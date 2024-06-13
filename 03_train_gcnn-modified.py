import os
import importlib
import argparse
import sys
import pathlib
import pickle
import numpy as np
from time import strftime
from shutil import copyfile
import gzip
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.eager as tfe

import utilities
from utilities import log

from utilities_tf import load_batch_gcnn


def load_batch_tf(x):
    return tf.py_func(
        load_batch_gcnn,
        [x],
        [tf.float32, tf.int32, tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.float32])


def pretrain(model, dataloader):
    """
    Pre-normalizes a model (i.e., PreNormLayer layers) over the given samples.

    Parameters
    ----------
    model : model.BaseModel
        A base model, which may contain some model.PreNormLayer layers.
    dataloader : tf.data.Dataset
        Dataset to use for pre-training the model.
    Return
    ------
    number of PreNormLayer layers processed.
    """
    model.pre_train_init()
    i = 0
    while True:
        for batch in dataloader:
            c, ei, ev, v, n_cs, n_vs, n_cands, cands, best_cands, cand_scores = batch
            batched_states = (c, ei, ev, v, n_cs, n_vs)

            if not model.pre_train(batched_states, tf.convert_to_tensor(True)):
                break

        res = model.pre_train_next()
        if res is None:
            break
        else:
            layer, name = res

        i += 1

    return i


def process(model, dataloader, top_k, optimizer=None):
    mean_loss = 0
    mean_kacc = np.zeros(len(top_k))

    n_samples_processed = 0
    for batch in dataloader:
        c, ei, ev, v, n_cs, n_vs, n_cands, cands, best_cands, cand_scores = batch
        batched_states = (c, ei, ev, v, tf.reduce_sum(n_cs, keepdims=True), tf.reduce_sum(n_vs, keepdims=True))  # prevent padding
        batch_size = len(n_cs.numpy())

        if optimizer:
            with tf.GradientTape() as tape:
                logits = model(batched_states, tf.convert_to_tensor(True)) # training mode
                logits = tf.expand_dims(tf.gather(tf.squeeze(logits, 0), cands), 0)  # filter candidate variables
                logits = model.pad_output(logits, n_cands.numpy())  # apply padding now
                loss = tf.losses.sparse_softmax_cross_entropy(labels=best_cands, logits=logits)
            grads = tape.gradient(target=loss, sources=model.variables)
            optimizer.apply_gradients(zip(grads, model.variables))
        else:
            logits = model(batched_states, tf.convert_to_tensor(False))  # eval mode
            logits = tf.expand_dims(tf.gather(tf.squeeze(logits, 0), cands), 0)  # filter candidate variables
            logits = model.pad_output(logits, n_cands.numpy())  # apply padding now
            loss = tf.losses.sparse_softmax_cross_entropy(labels=best_cands, logits=logits)

        true_scores = model.pad_output(tf.reshape(cand_scores, (1, -1)), n_cands)
        true_bestscore = tf.reduce_max(true_scores, axis=-1, keepdims=True)
        true_scores = true_scores.numpy()
        true_bestscore = true_bestscore.numpy()

        kacc = []
        for k in top_k:
            pred_top_k = tf.nn.top_k(logits, k=k)[1].numpy()
            pred_top_k_true_scores = np.take_along_axis(true_scores, pred_top_k, axis=1)
            kacc.append(np.mean(np.any(pred_top_k_true_scores == true_bestscore, axis=1)))
        kacc = np.asarray(kacc)

        mean_loss += loss.numpy() * batch_size
        mean_kacc += kacc * batch_size
        n_samples_processed += batch_size

    mean_loss /= n_samples_processed
    mean_kacc /= n_samples_processed

    return mean_loss, mean_kacc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset','mixed6','mixed5',"mixed_gisp0.3","mixed_gisp0.5","mixed_sm1","mixed_sm1h",
                 "gisp_0.3_n125","CFLP_gasse120","setcover-gasse-1200",
                "gisp_0.3_n100","gisp_0.5_n75","CA-gasse-l", "setcover-gasse-m","indset-gasse-m","CFLP_gasse","GISP_GNP_easy2","nn_verification","gisp_0.3_n120","gisp_0.3_n110","energy_small_10day","GISP_GNP_easyLP","GISP_GNP_easy125","GISP_GNP_easy120","mixed_GISP_GNP","GISP_GNP_easy115","GISP_GNP_easy110","mixed_GISP_GNP115","mixed_GISP_GNP110","energy_small_15day","mixed_GISP_energy","energy_small_16day","mixed_5_domain","water_mixed",
                "mmtcInst/sm1","mmtcInst/sm1h","GISP_GNP_easy-LP_format","water_network","water2"],#to add: small MVC instances, mmtcInst/mmtc1
    )
    parser.add_argument(
        '-m', '--model',
        help='GCNN model to be trained.',
        type=str,
        default='baseline',
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=utilities.valid_seed,
        default=0,
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--comb',
        help='which combinations',
        type=int
    )
    parser.add_argument(
        '--n_samples_per_domain',
        help='which combinations',
        type=int
    )
    parser.add_argument(
        '--n_samples_per_instance',
        help='how many state-action pairs to use per instance',
        type=int
    )


    args = parser.parse_args()

    ### HYPER PARAMETERS ###
    max_epochs = 1000
    epoch_size = 312
    batch_size = 32
    pretrain_batch_size = 128
    valid_batch_size = 128
    lr = 0.001
    patience = 10
    early_stopping = 20
    top_k = [1, 3, 5, 10]
    train_ncands_limit = np.inf
    valid_ncands_limit = np.inf

    problem_folders = {
        'setcover': 'setcover/weimin',
        'cauctions': 'cauctions/100_500',
        'facilities': 'facilities/100_100_5',
        'indset': 'indset/500_4',
    }
    
            
    
    print("comb",args.comb)
    
    if "mixed" in args.problem:
        if not (args.comb==None):
            if (args.n_samples_per_instance==None):
                running_dir = f"trained_models_new/{args.problem}_no{args.comb}/{args.model}/{args.seed}"
            else:
                running_dir = f"trained_models_new/{args.problem}_no{args.comb}/{args.model}/{args.seed}"
        else:
            if (args.n_samples_per_instance==None):
                running_dir = f"trained_models_new/{args.problem}_n{args.n_samples_per_domain}/{args.model}/{args.seed}"
            else:
                running_dir = f"trained_models_new/{args.problem}_n{args.n_samples_per_domain}_samples{args.n_samples_per_instance}/{args.model}/{args.seed}"
            #{problem_folder}_n{args.n_samples_per_domain}_samples{args.n_samples_per_instance}
    else:
        if "mmtc" in args.problem:
            new=(args.problem).replace("/","_")
            running_dir = f"trained_models_new/{new}_n{args.n_samples_per_domain}/{args.model}/{args.seed}"
        #problem_folder = problem_folders[args.problem]
        else:
            if (args.n_samples_per_instance==None):
                running_dir = f"trained_models_new/{args.problem}_n{args.n_samples_per_domain}/{args.model}/{args.seed}"
            else:
                running_dir = f"trained_models_new/{args.problem}_n{args.n_samples_per_domain}_samples{args.n_samples_per_instance}/{args.model}/{args.seed}"

    os.makedirs(running_dir)

    ### LOG ###
    logfile = os.path.join(running_dir, 'log.txt')

    log(f"max_epochs: {max_epochs}", logfile)
    log(f"epoch_size: {epoch_size}", logfile)
    log(f"batch_size: {batch_size}", logfile)
    log(f"pretrain_batch_size: {pretrain_batch_size}", logfile)
    log(f"valid_batch_size : {valid_batch_size }", logfile)
    log(f"lr: {lr}", logfile)
    log(f"patience : {patience }", logfile)
    log(f"early_stopping : {early_stopping }", logfile)
    log(f"top_k: {top_k}", logfile)
    log(f"problem: {args.problem}", logfile)
    log(f"gpu: {args.gpu}", logfile)
    log(f"seed {args.seed}", logfile)

    ### NUMPY / TENSORFLOW SETUP ###
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.enable_eager_execution(config)
    tf.executing_eagerly()

    rng = np.random.RandomState(args.seed)
    tf.set_random_seed(rng.randint(np.iinfo(int).max))

    ### SET-UP DATASET ###
    if args.problem=="mixed5":
        all_domain =["setcover-gasse-m","indset-gasse-m","CFLP_gasse","GISP_GNP_easy2","CA-gasse-m"]
    if args.problem=="mixed6":
        all_domain =["setcover-gasse-m","indset-gasse-m","CFLP_gasse","GISP_GNP_easy2","nn_verification", "CA-gasse-m"]
    if args.problem=="mixed_gisp0.3":
        all_domain =["CA-gasse-l", "setcover-gasse-m","indset-gasse-m","CFLP_gasse","gisp_0.3_n100"]
    if args.problem=="mixed_gisp0.5":
        all_domain =["CA-gasse-l", "setcover-gasse-m","indset-gasse-m","CFLP_gasse","gisp_0.5_n75"]
    if args.problem=="mixed_sm1":
        all_domain =["CA-gasse-l", "setcover-gasse-m","indset-gasse-m","CFLP_gasse","mmtcInst/sm1"]
    if args.problem=="mixed_sm1h":
        all_domain =["CA-gasse-l", "setcover-gasse-m","indset-gasse-m","CFLP_gasse","mmtcInst/sm1h"]
    if args.problem=="mixed_GISP_GNP":
        all_domain =["setcover-gasse-1200","CFLP_gasse120","CA-gasse-l","indset-gasse-m","GISP_GNP_easyLP"]
    if args.problem=="mixed_GISP_GNP110":
        all_domain =["CA-gasse-l", "setcover-gasse-m","indset-gasse-m","CFLP_gasse","GISP_GNP_easy110"]
    if args.problem=="mixed_GISP_GNP115":
        all_domain =["CA-gasse-l", "setcover-gasse-m","indset-gasse-m","CFLP_gasse","GISP_GNP_easy115"]
    if args.problem=="mixed_GISP_energy":
        all_domain =["energy_small_16day","CA-gasse-l", "setcover-gasse-m","indset-gasse-m","CFLP_gasse","GISP_GNP_easy115"]
    if args.problem=="mixed_5_domain":
        all_domain =["CA-gasse-l", "setcover-gasse-m","indset-gasse-m","CFLP_gasse","GISP_GNP_easy115"]
    if args.problem=="water_mixed":
        all_domain =["CA-gasse-l", "setcover-gasse-m","indset-gasse-m","CFLP_gasse","GISP_GNP_easy115","water2"]

    np.random.seed(args.seed)
    if "mixed" in args.problem:
        if not (args.comb==None):
            use_files = list(set(all_domain)-set([all_domain[args.comb]]))
        else:
            use_files = list(set(all_domain))
        
        train_files=[]
        #shard = list(range(0,args.n_samples_per_domain))
        #shard=np.random.permutation(np.arange(1000))[:args.n_samples_per_domain]

        f_names_L=[]
        if (args.n_samples_per_instance==None):
        
            for problem_folder in use_files:
                if problem_folder in ["water_network","water2"]:
                    shard = list(range(0,len(list(pathlib.Path(f'/project/dilkina_438/weiminhu/miplib/learn2branch/data/instances_sep/{problem_folder}').glob('train*')))))
                    for ind in shard:
                        train_files += list(pathlib.Path(f'data/samples_sep/{problem_folder}/train{ind}').glob('sample_*.pkl'))   
                else:
                    my_file=f"trained_models_new/{problem_folder}_n{args.n_samples_per_domain}/{args.model}/{args.seed}/data_info.txt"
                    #with open("/project/dilkina_438/weiminhu/miplib/learn2branch/trained_models_new/GISP_GNP_easy115_n80/baseline/2/data_info.txt", "r") as grilled_cheese:
                    with open(my_file, "r") as grilled_cheese:
                        lines = grilled_cheese.readlines()
                    shard=lines[1:-2]
                    shard=[item.replace("\n","") for item in shard]
                    f_names_L.append(my_file)
                    for ind in shard:
                        train_files += list(pathlib.Path(f'data/samples_sep/{problem_folder}/train{ind}').glob('sample_*.pkl'))
            
        else:
            for problem_folder in use_files:
                if problem_folder in ["water_network","water2"]:
                    shard = list(range(0,len(list(pathlib.Path(f'/project/dilkina_438/weiminhu/miplib/learn2branch/data/instances_sep/{args.problem}').glob('train*')))))
                    for ind in shard:
                        train_files += list(pathlib.Path(f'data/samples_sep/{problem_folder}/train{ind}').glob('sample_*.pkl')) 
                else:
                    my_file=f"trained_models_new/{problem_folder}_n{args.n_samples_per_domain}_samples{args.n_samples_per_instance}/{args.model}/{args.seed}/used_data.txt"
                    #with open("/project/dilkina_438/weiminhu/miplib/learn2branch/trained_models_new/GISP_GNP_easy115_n80/baseline/2/data_info.txt", "r") as grilled_cheese:
                    with open(my_file, "r") as grilled_cheese:
                        lines = grilled_cheese.readlines()
                    for f_name in lines:
                        train_files += lines
        with open(os.path.join(running_dir, 'data_info.txt'), 'w') as f:
            f.write(f"Trained on domains:\n")
            for line in use_files:
                f.write(f"{line}\n")
            f.write(f"Per domain list:\n")
            for name in f_names_L:
                f.write(f"{name}\n")
            f.write(f"Total number of training samples: {args.problem}\n")
            f.write(f"{len(train_files)}\n")
        valid_files=[]
        
        for problem_folder in use_files:
            shard = list(range(0,100))
            if args.problem in ["water_network","water2","water_mixed"]:
                shard = list(range(0,len(list(pathlib.Path(f'/project/dilkina_438/weiminhu/miplib/learn2branch/data/instances_sep/{problem_folder}').glob('val*')))))
            for ind in shard:
                valid_files += list(pathlib.Path(f'data/samples_sep/{problem_folder}/val{ind}').glob('sample_*.pkl'))      
    else:
        train_files=[]
        #shard = list(range(0,args.n_samples_per_domain))
        shard=np.random.permutation(np.arange(800))[:args.n_samples_per_domain]
        if args.problem in ["water_network","water2"]:
            shard = list(range(0,len(list(pathlib.Path(f'/project/dilkina_438/weiminhu/miplib/learn2branch/data/instances_sep/{args.problem}').glob('train*')))))
        for ind in shard:
            if (args.n_samples_per_instance==None):
                # 10 samples per instance
                train_files += list(pathlib.Path(f'data/samples_sep/{args.problem}/train{ind}').glob('sample_*.pkl'))
            else:
                # 5 samples per instance
                temp = list(np.random.permutation(list(pathlib.Path(f'data/samples_sep/{args.problem}/train{ind}').glob('sample_*.pkl')))[:5])
                train_files += temp
        print("total number of training samples:",len(train_files))
        print("training shards:",shard)
        with open(os.path.join(running_dir, 'data_info.txt'), 'w') as f:
            f.write(f"Trained on domain: {args.problem}, shards:\n")
            for ind in shard:
                f.write(f"train{ind}\n")
            f.write(f"Total number of training samples: {args.problem}\n")
            f.write(f"{len(train_files)}\n")
        with open(os.path.join(running_dir, 'used_data.txt'), 'w') as f:
            for f_name in train_files:
                f.write(f"{f_name}\n")
            f.write(f"Total number of training samples: {args.problem}\n")
            f.write(f"{len(train_files)}\n")
        valid_files=[]
        #shard = list(range(0,100))
        shard = list(range(0,100))
        if args.problem in ["water_network","water2"]:
            shard = list(range(0,len(list(pathlib.Path(f'/project/dilkina_438/weiminhu/miplib/learn2branch/data/instances_sep/{args.problem}').glob('val*')))))
        for ind in shard:
            valid_files += list(pathlib.Path(f'data/samples_sep/{args.problem}/val{ind}').glob('sample_*.pkl'))  
        #train_files = list(pathlib.Path(f'data/samples_sep/{problem_folder}/train').glob('sample_*.pkl'))
        #valid_files = list(pathlib.Path(f'data/samples_sep/{problem_folder}/valid').glob('sample_*.pkl'))
    #print("check train files1")
    #print(train_files)

    # reset?
    rng = np.random.RandomState(args.seed)
    tf.set_random_seed(rng.randint(np.iinfo(int).max))


    def take_subset(sample_files, cands_limit):
        nsamples = 0
        ncands = 0
        for filename in sample_files:
            with gzip.open(filename, 'rb') as file:
                sample = pickle.load(file)

            _, _, _, cands, _ = sample['data']
            ncands += len(cands)
            nsamples += 1

            if ncands >= cands_limit:
                log(f"  dataset size limit reached ({cands_limit} candidate variables)", logfile)
                break

        return sample_files[:nsamples]


    if train_ncands_limit < np.inf:
        train_files = take_subset(rng.permutation(train_files), train_ncands_limit)
    log(f"{len(train_files)} training samples", logfile)
    if valid_ncands_limit < np.inf:
        valid_files = take_subset(valid_files, valid_ncands_limit)
    log(f"{len(valid_files)} validation samples", logfile)

    train_files = [str(x) for x in train_files]
    #if args.problem in ["water_network","water2","water_mixed"]:
    #    train_files = train_files[:(1632*5)]
    
    valid_files = [str(x) for x in valid_files]
    #if args.problem in ["water_network","water2","water_mixed"]:
    #    valid_files = valid_files[:]
    print("number of training samples")
    print(len(train_files))
    print("number of val samples")
    print(len(valid_files))
    valid_data = tf.data.Dataset.from_tensor_slices(valid_files)
    valid_data = valid_data.batch(valid_batch_size)
    valid_data = valid_data.map(load_batch_tf)
    valid_data = valid_data.prefetch(1)

    pretrain_files = [f for i, f in enumerate(train_files) if i % 10 == 0]
    pretrain_data = tf.data.Dataset.from_tensor_slices(pretrain_files)
    pretrain_data = pretrain_data.batch(pretrain_batch_size)
    pretrain_data = pretrain_data.map(load_batch_tf)
    pretrain_data = pretrain_data.prefetch(1)

    # check
    from tensorflow.python.client import device_lib
    print("using GPU?????",device_lib.list_local_devices())

    ### MODEL LOADING ###
    sys.path.insert(0, os.path.abspath(f'models/{args.model}'))
    import model
    importlib.reload(model)
    model = model.GCNPolicy()
    del sys.path[0]

    L_train_loss=[]
    L_val_loss=[]
    L_epoch=[]

    ### TRAINING LOOP ###
    optimizer = tf.train.AdamOptimizer(learning_rate=lambda: lr)  # dynamic LR trick
    best_loss = np.inf
    for epoch in range(max_epochs + 1):
        log(f"EPOCH {epoch}...", logfile)
        epoch_loss_avg = tfe.metrics.Mean()
        epoch_accuracy = tfe.metrics.Accuracy()

        # TRAIN
        if epoch == 0:
            n = pretrain(model=model, dataloader=pretrain_data)
            log(f"PRETRAINED {n} LAYERS", logfile)
            # model compilation
            model.call = tfe.defun(model.call, input_signature=model.input_signature)
            train_loss=None
        else:
            # bugfix: tensorflow's shuffle() seems broken...
            epoch_train_files = rng.choice(train_files, epoch_size * batch_size, replace=True)
            train_data = tf.data.Dataset.from_tensor_slices(epoch_train_files)
            train_data = train_data.batch(batch_size)
            train_data = train_data.map(load_batch_tf)
            train_data = train_data.prefetch(1)
            train_loss, train_kacc = process(model, train_data, top_k, optimizer)
            log(f"TRAIN LOSS: {train_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, train_kacc)]), logfile)

        # TEST
        valid_loss, valid_kacc = process(model, valid_data, top_k, None)
        log(f"VALID LOSS: {valid_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, valid_kacc)]), logfile)

        L_train_loss.append(train_loss)
        L_val_loss.append(valid_loss)
        L_epoch.append(epoch)

        if valid_loss < best_loss:
            plateau_count = 0
            best_loss = valid_loss
            model.save_state(os.path.join(running_dir, 'best_params.pkl'))
            log(f"  best model so far", logfile)
        else:
            plateau_count += 1
            if plateau_count % early_stopping == 0:
                log(f"  {plateau_count} epochs without improvement, early stopping", logfile)
                break
            if plateau_count % patience == 0:
                lr *= 0.2
                log(f"  {plateau_count} epochs without improvement, decreasing learning rate to {lr}", logfile)

        

        L_train_loss.append(train_loss)
        L_val_loss.append(valid_loss)
        L_epoch.append(epoch)

    model.restore_state(os.path.join(running_dir, 'best_params.pkl'))
    valid_loss, valid_kacc = process(model, valid_data, top_k, None)
    log(f"BEST VALID LOSS: {valid_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, valid_kacc)]), logfile)
    df=pd.DataFrame({"train_loss":L_train_loss,"val_loss": L_val_loss,"epoch": L_epoch})
    df.to_csv(os.path.join(running_dir, 'training_curve.csv'))

