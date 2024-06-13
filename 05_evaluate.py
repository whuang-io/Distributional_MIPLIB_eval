import os
import sys
import importlib
import argparse
import csv
import numpy as np
import time
import pickle
import pandas as pd
import pathlib

import pyscipopt as scip
from pyscipopt import Eventhdlr, SCIP_RESULT, SCIP_EVENTTYPE
import tensorflow as tf
import tensorflow.contrib.eager as tfe

import svmrank

import utilities



class PolicyBranching(scip.Branchrule):

    def __init__(self, policy):
        super().__init__()

        self.policy_type = policy['type']
        self.policy_name = policy['name']

        if self.policy_type == 'gcnn':
            model = policy['model']
            model.restore_state(policy['parameters'])
            self.policy = tfe.defun(model.call, input_signature=model.input_signature)

        elif self.policy_type == 'internal':
            self.policy = policy['name']

        elif self.policy_type == 'ml-competitor':
            self.policy = policy['model']

            # feature parameterization
            self.feat_shift = policy['feat_shift']
            self.feat_scale = policy['feat_scale']
            self.feat_specs = policy['feat_specs']

        else:
            raise NotImplementedError

    def branchinitsol(self):
        self.ndomchgs = 0
        self.ncutoffs = 0
        self.state_buffer = {}
        self.khalil_root_buffer = {}
        self.ml_time=0
        self.ml_time_feat=0
        self.ml_time_forward=0
        self.ml_time_rank=0

    def branchexeclp(self, allowaddcons):

        # SCIP internal branching rule
        if self.policy_type == 'internal':
            result = self.model.executeBranchRule(self.policy, allowaddcons)

        # custom policy branching
        else:
            
            candidate_vars, *_ = self.model.getPseudoBranchCands()
            candidate_mask = [var.getCol().getLPPos() for var in candidate_vars]

            # initialize root buffer for Khalil features extraction
            if self.model.getNNodes() == 1 \
                    and self.policy_type == 'ml-competitor' \
                    and self.feat_specs['type'] in ('khalil', 'all'):
                utilities.extract_khalil_variable_features(self.model, [], self.khalil_root_buffer)

            if len(candidate_vars) == 1:
                best_var = candidate_vars[0]

            elif self.policy_type == 'gcnn':
                start=time.time()
                state = utilities.extract_state(self.model, self.state_buffer)
                t1=time.time()
                self.ml_time_feat += (t1-start)

                # convert state to tensors
                c, e, v = state
                state = (
                    tf.convert_to_tensor(c['values'], dtype=tf.float32),
                    tf.convert_to_tensor(e['indices'], dtype=tf.int32),
                    tf.convert_to_tensor(e['values'], dtype=tf.float32),
                    tf.convert_to_tensor(v['values'], dtype=tf.float32),
                    tf.convert_to_tensor([c['values'].shape[0]], dtype=tf.int32),
                    tf.convert_to_tensor([v['values'].shape[0]], dtype=tf.int32),
                )

                var_logits = self.policy(state, tf.convert_to_tensor(False)).numpy().squeeze(0)

                t2=time.time()
                self.ml_time_forward += (t2-t1)
                candidate_scores = var_logits[candidate_mask]
                best_var = candidate_vars[candidate_scores.argmax()]
                end=time.time()
                self.ml_time_rank += (end-t2)
                self.ml_time += (end-start)

            elif self.policy_type == 'ml-competitor':

                # build candidate features
                candidate_states = []
                if self.feat_specs['type'] in ('all', 'gcnn_agg'):
                    state = utilities.extract_state(self.model, self.state_buffer)
                    candidate_states.append(utilities.compute_extended_variable_features(state, candidate_mask))
                if self.feat_specs['type'] in ('all', 'khalil'):
                    candidate_states.append(utilities.extract_khalil_variable_features(self.model, candidate_vars, self.khalil_root_buffer))
                candidate_states = np.concatenate(candidate_states, axis=1)

                # feature preprocessing
                candidate_states = utilities.preprocess_variable_features(candidate_states, self.feat_specs['augment'], self.feat_specs['qbnorm'])

                # feature normalization
                candidate_states =  (candidate_states - self.feat_shift) / self.feat_scale

                candidate_scores = self.policy.predict(candidate_states)
                best_var = candidate_vars[candidate_scores.argmax()]

            else:
                raise NotImplementedError

            self.model.branchVar(best_var)
            result = scip.SCIP_RESULT.BRANCHED

        # fair node counting
        if result == scip.SCIP_RESULT.REDUCEDDOM:
            self.ndomchgs += 1
        elif result == scip.SCIP_RESULT.CUTOFF:
            self.ncutoffs += 1

        return {'result': result}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     'problem',
    #     help='MILP instance type to process.',
    #     choices=['setcover', 'cauctions', 'facilities', 'indset','mixed6','mixed5'],
    # )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--model_folder',
        help='which model folder',
        type=str,
    )
    parser.add_argument(
        '--save_folder',
        help='which folder to save results to',
        type=str,
    )
    parser.add_argument(
        "--ML",
        help="whether this is ML",
        default=False,
        type=lambda x: (str(x).lower() == "true"),
    )
    parser.add_argument(
        '--domain',
        help='which domain',
        type=str,
    )
    parser.add_argument(
        '--seed',
        help='random seed',
        type=int,
    )
    parser.add_argument(
        '--part',
        help='which part',
        default=0,
        type=int,
    )
    args = parser.parse_args()

    
    instances = []
    # seeds = [0, 1, 2, 3, 4]
    seeds = [args.seed] # one seed for now
    gcnn_models = ['baseline']
    #other_models = ['extratrees_gcnn_agg', 'lambdamart_khalil', 'svmrank_khalil']
    internal_branchers = ['relpscost']
    #time_limit = 3600
    time_limit = 800

    formats = ["lp","mps"]

    # if args.problem == 'setcover':
    #     instances += [{'type': 'small', 'path': f"data/instances/setcover/transfer_500r_1000c_0.05d/instance_{i+1}.lp"} for i in range(20)]
    #     instances += [{'type': 'medium', 'path': f"data/instances/setcover/transfer_1000r_1000c_0.05d/instance_{i+1}.lp"} for i in range(20)]
    #     instances += [{'type': 'big', 'path': f"data/instances/setcover/transfer_2000r_1000c_0.05d/instance_{i+1}.lp"} for i in range(20)]
    #     gcnn_models += ['mean_convolution', 'no_prenorm']

    # elif args.problem == 'cauctions':
    #     instances += [{'type': 'small', 'path': f"data/instances/cauctions/transfer_100_500/instance_{i+1}.lp"} for i in range(20)]
    #     instances += [{'type': 'medium', 'path': f"data/instances/cauctions/transfer_200_1000/instance_{i+1}.lp"} for i in range(20)]
    #     instances += [{'type': 'big', 'path': f"data/instances/cauctions/transfer_300_1500/instance_{i+1}.lp"} for i in range(20)]

    # elif args.problem == 'facilities':
    #     instances += [{'type': 'small', 'path': f"data/instances/facilities/transfer_100_100_5/instance_{i+1}.lp"} for i in range(20)]
    #     instances += [{'type': 'medium', 'path': f"data/instances/facilities/transfer_200_100_5/instance_{i+1}.lp"} for i in range(20)]
    #     instances += [{'type': 'big', 'path': f"data/instances/facilities/transfer_400_100_5/instance_{i+1}.lp"} for i in range(20)]

    # elif args.problem == 'indset':
    #     instances += [{'type': 'small', 'path': f"data/instances/indset/transfer_500_4/instance_{i+1}.lp"} for i in range(20)]
    #     instances += [{'type': 'medium', 'path': f"data/instances/indset/transfer_1000_4/instance_{i+1}.lp"} for i in range(20)]
    #     instances += [{'type': 'big', 'path': f"data/instances/indset/transfer_1500_4/instance_{i+1}.lp"} for i in range(20)]
        
    # elif args.problem in ['mixed5','mixed6']:
    dest = "/project/dilkina_438/weiminhu/miplib/learn2branch/data/instances_sep"
    
    temp_L=[]
    if args.domain in ["water_network","water2"]:
        high = len(list(pathlib.Path(f'/project/dilkina_438/weiminhu/miplib/learn2branch/data/instances_sep/{args.domain}').glob('test*')))
        low=0
    elif args.domain in ["SetCovering_CL-LNS_LB-RELAX/5000_4000","INDSET_CL-LNS_mps_format/6000_erdos_renyi_LP_format","CA_CL-LNS/2000_4000"]:
        if args.part==0:
            high=25
            low=0
            # if args.domain in ["INDSET_CL-LNS_mps_format/6000_erdos_renyi_LP_format"]:
            #     low=low+15
            # for scip-indset CL-LNS: low=low+15
        if args.part==1:
            high=50
            low=25
            # if args.domain in ["INDSET_CL-LNS_mps_format/6000_erdos_renyi_LP_format"]:
            #     low=low+15
            #high=low+15
        if args.part==2:
            high=75
            low=50
            # if args.domain in ["INDSET_CL-LNS_mps_format/6000_erdos_renyi_LP_format"]:
            #     low=low+15
            #high=low+15
        if args.part==3:
            high=100
            low=75
            # if args.domain in ["INDSET_CL-LNS_mps_format/6000_erdos_renyi_LP_format"]:
            #     low=low+15
            #high=low+15
        low=low+22
        time_limit = 3600
    else:
        high=100
        low=0
    for shard in list(range(low,high)):
        temp_L += [f"data/instances_sep/{args.domain}/test{shard}/{item}" for item in os.listdir(os.path.join(dest,args.domain,"test"+str(shard))) if (item.split(".")[-1] in formats)]
    instances += [{'type': args.domain, 'path': item} for item in temp_L]

    # else:
    #     raise NotImplementedError

    branching_policies = []

    # # ML baselines
    # for model in other_models:
    #     for seed in seeds:
    #         branching_policies.append({
    #             'type': 'ml-competitor',
    #             'name': model,
    #             'seed': seed,
    #             'model': f'trained_models/{args.problem}/{model}/{seed}',
    #         })
    # GCNN models
    # for model in gcnn_models:
    #     for seed in seeds:
    #         branching_policies.append({
    #             'type': 'gcnn',
    #             'name': model,
    #             'seed': seed,
    #             'parameters': f'trained_models/{args.problem}/{model}/{seed}/best_params.pkl'
    #         })

    if args.ML:
        if "/" in args.domain:
            temp=args.domain.replace("/","_")
            result_file = f"{args.model_folder}_{temp}_{time.strftime('%Y%m%d-%H%M%S')}_train_seed{args.seed}.csv"
        else:
            temp=args.domain
            result_file = f"{args.model_folder}_{args.domain}_{time.strftime('%Y%m%d-%H%M%S')}_train_seed{args.seed}.csv"
        for model in gcnn_models:
            for seed in seeds:
                branching_policies.append({
                    'type': 'gcnn',
                    'name': model,
                    'seed': seed,
                    'parameters': f'trained_models_new/{args.model_folder}/{model}/{args.seed}/best_params.pkl',
                })
    else:
        if "/" in args.domain:
            temp=args.domain.replace("/","_")
            result_file = f"SCIP_internal_relpscost_{temp}_{time.strftime('%Y%m%d-%H%M%S')}_seed{args.seed}.csv"
        else:
            temp=args.domain
            result_file = f"SCIP_internal_relpscost_{args.domain}_{time.strftime('%Y%m%d-%H%M%S')}_seed{args.seed}.csv"
        # SCIP internal brancher baselines
        for brancher in internal_branchers:
            for seed in seeds:
                branching_policies.append({
                        'type': 'internal',
                        'name': brancher,
                        'seed': seed,
                 })
        

    print(f"domain: {args.domain}")
    print(f"gpu: {args.gpu}")
    print(f"time limit: {time_limit} s")

    ### TENSORFLOW SETUP ###
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.enable_eager_execution(config)
    tf.executing_eagerly()

    # load and assign tensorflow models to policies (share models and update parameters)
    loaded_models = {}
    for policy in branching_policies:
        if policy['type'] == 'gcnn':
            if policy['name'] not in loaded_models:
                sys.path.insert(0, os.path.abspath(f"models/{policy['name']}"))
                import model
                importlib.reload(model)
                loaded_models[policy['name']] = model.GCNPolicy()
                del sys.path[0]
            policy['model'] = loaded_models[policy['name']]

    # # load ml-competitor models
    # for policy in branching_policies:
    #     if policy['type'] == 'ml-competitor':
    #         try:
    #             with open(f"{policy['model']}/normalization.pkl", 'rb') as f:
    #                 policy['feat_shift'], policy['feat_scale'] = pickle.load(f)
    #         except:
    #             policy['feat_shift'], policy['feat_scale'] = 0, 1

    #         with open(f"{policy['model']}/feat_specs.pkl", 'rb') as f:
    #             policy['feat_specs'] = pickle.load(f)

    #         if policy['name'].startswith('svmrank'):
    #             policy['model'] = svmrank.Model().read(f"{policy['model']}/model.txt")
    #         else:
    #             with open(f"{policy['model']}/model.pkl", 'rb') as f:
    #                 policy['model'] = pickle.load(f)

    print("running SCIP...")

    fieldnames = [
        'policy',
        'seed',
        'type',
        'instance',
        'nnodes',
        'nlps',
        'stime',
        'gap',
        'status',
        'ndomchgs',
        'ncutoffs',
        'walltime',
        'proctime',
        'integral',
        'integral2',
        'integral_n1',
        'integral_n2',
        'ml_time',
        'ml_time_feat',
        'ml_time_forward',
        'ml_time_rank'
    ]
    os.makedirs('results', exist_ok=True)
    os.makedirs(f"results/{args.save_folder}", exist_ok=True)

    with open(f"results/{args.save_folder}/{result_file}", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for instance in instances:
            print(f"{instance['type']}: {instance['path']}...")

            for policy in branching_policies:
                tf.set_random_seed(policy['seed'])

                m = scip.Model()
                L_t = []
                L_objbst = []
                L_objbnd = []
                L_gap = []
                integral=[0]
                integral2=[0]
                integral_n1=[0]
                integral_n2=[0]
                L_n1=[]
                L_n2=[]
                
                class NodeSolvedEvent(Eventhdlr):
                    def eventinit(self):
                        self.model.catchEvent(SCIP_EVENTTYPE.LPSOLVED, self)
                        self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)
                        self.sense=m.getObjectiveSense()
                
                    def eventexit(self):
                        self.model.dropEvent(SCIP_EVENTTYPE.LPSOLVED, self)
                        self.model.dropEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

                        objbst = self.model.getPrimalbound()
                        objbnd = self.model.getDualbound()
                        #gap = abs((objbst - objbnd) / objbst)
                        cur_time = self.model.getSolvingTime()
                        cur_node1 = self.model.getNNodes()
                        #cur_node2 = self.model.getNTotalNodes()
                        cur_node2 = 0
                        
                        L_objbst.append(objbst)
                        L_objbnd.append(objbnd)
                        L_t.append(cur_time)
                        L_n1.append(cur_node1)
                        L_n2.append(cur_node2)
                        if (objbst==0) and (objbnd==0):
                            gap_=0
                        elif (objbst==0) and (not (objbnd==0)):
                            gap_=1
                        else:
                            gap_=abs(objbst - objbnd) / abs(objbst)
                        L_gap.append(gap_)
                
                    def eventexec(self, event):
                        
                        objbst = self.model.getPrimalbound()
                        objbnd = self.model.getDualbound()
                        #gap = abs((objbst - objbnd) / objbst)
                        cur_time = self.model.getSolvingTime()
                        cur_node1 = self.model.getNNodes()
                        #cur_node2 = self.model.getNTotalNodes()
                        cur_node2 = 0
                        
                        L_objbst.append(objbst)
                        L_objbnd.append(objbnd)
                        L_t.append(cur_time)
                        L_n1.append(cur_node1)
                        L_n2.append(cur_node2)
                        if (objbst==0) and (objbnd==0):
                            gap_=0
                        elif (objbst==0) and (not (objbnd==0)):
                            gap_=1
                        else:
                            gap_=abs(objbst - objbnd) / abs(objbst)
                        L_gap.append(gap_)
                            
                        if len(L_t)>1:
                           delta_t = L_t[-1]-L_t[-2]
                           delta_n1= L_n1[-1]-L_n1[-2]
                           delta_n2= L_n2[-1]-L_n2[-2]
                        else:
                           delta_t = L_t[-1]
                           delta_n1= L_n1[-1]
                           delta_n2= L_n2[-1]

                        if cur_time>2:
                            if self.sense=="minimize":
                                integral2[0] += -(objbnd - objbst) * delta_t
                            else:
                                integral2[0] += (objbnd - objbst) * delta_t
                            integral[0] += delta_t * L_gap[-1]
                            integral_n1[0]+= delta_n1 * L_gap[-1]
                            integral_n2[0] += delta_n2 * L_gap[-1]
                        
                eventhdlr = NodeSolvedEvent()
                m.includeEventhdlr(eventhdlr, "primal_dual_events", "python event handler to catch LPSOLVED and BESTSOLFOUND events")
                
                m.setIntParam('display/verblevel', 0)
                m.readProblem(f"{instance['path']}")
                utilities.init_scip_params(m, seed=policy['seed'])
                m.setIntParam('timing/clocktype', 1)  # 1: CPU user seconds, 2: wall clock time
                m.setRealParam('limits/time', time_limit)

                if "mmtc" in args.domain:
                    m.setRealParam('limits/gap', 0.005)

                brancher = PolicyBranching(policy)
                m.includeBranchrule(
                    branchrule=brancher,
                    name=f"{policy['type']}:{policy['name']}",
                    desc=f"Custom PySCIPOpt branching policy.",
                    priority=666666, maxdepth=-1, maxbounddist=1)

                walltime = time.perf_counter()
                proctime = time.process_time()

                m.optimize()

                walltime = time.perf_counter() - walltime
                proctime = time.process_time() - proctime

                stime = m.getSolvingTime()
                nnodes = m.getNNodes()
                nlps = m.getNLPs()
                gap = m.getGap()
                status = m.getStatus()
                ndomchgs = brancher.ndomchgs
                ncutoffs = brancher.ncutoffs

                ml_time = brancher.ml_time
                ml_time_feat = brancher.ml_time_feat
                ml_time_forward = brancher.ml_time_forward
                ml_time_rank = brancher.ml_time_rank

                writer.writerow({
                    'policy': f"{policy['type']}:{policy['name']}",
                    'seed': policy['seed'],
                    'type': instance['type'],
                    'instance': instance['path'],
                    'nnodes': nnodes,
                    'nlps': nlps,
                    'stime': stime,
                    'gap': gap,
                    'status': status,
                    'ndomchgs': ndomchgs,
                    'ncutoffs': ncutoffs,
                    'walltime': walltime,
                    'proctime': proctime,
                    'integral': integral[0],
                    'integral2': integral2[0],
                    'integral_n1':integral_n1[0],
                    'integral_n2':integral_n2[0],
                    'ml_time': ml_time,
                    'ml_time_feat': ml_time_feat,
                    'ml_time_forward': ml_time_forward,
                    'ml_time_rank': ml_time_rank
                })

                csvfile.flush()
                m.freeProb()

                res={'t':L_t, 'objbst':L_objbst, 'objbnd':L_objbnd, 'gap': L_gap, 'n1':L_n1, 'n2':L_n2}
                df_res=pd.DataFrame.from_dict(res)
                f_name = os.path.basename(instance['path']).split(".")[-2]+".csv"
                if not os.path.exists(f"results/{args.save_folder}/full_into"):
                    os.mkdir(f"results/{args.save_folder}/full_into")
                if not os.path.exists(f"results/{args.save_folder}/full_into/{temp}"):
                    os.mkdir(f"results/{args.save_folder}/full_into/{temp}")
                if not os.path.exists(f"results/{args.save_folder}/full_into/{temp}/{args.model_folder}"):
                    os.mkdir(f"results/{args.save_folder}/full_into/{temp}/{args.model_folder}")
                if not os.path.exists(f"results/{args.save_folder}/full_into/{temp}/{args.model_folder}/{args.seed}"):
                    os.mkdir(f"results/{args.save_folder}/full_into/{temp}/{args.model_folder}/{args.seed}")
    
                df_res.to_csv(f"results/{args.save_folder}/full_into/{temp}/{args.model_folder}/{args.seed}/{f_name}")

                del eventhdlr
                del L_t
                del L_objbst
                del L_objbnd
                del L_gap
                del integral
                del integral2
                del integral_n1
                del integral_n2

                print(f"  {policy['type']}:{policy['name']} {policy['seed']} - {nnodes} ({nnodes+2*(ndomchgs+ncutoffs)}) nodes {nlps} lps {stime:.2f} ({walltime:.2f} wall {proctime:.2f} proc) s. {status}")

