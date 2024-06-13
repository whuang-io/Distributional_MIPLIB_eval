# Distributional_MIPLIB_eval

Evaluation code used in experiments on Distributional MIPLIB. The code is modified based on the Learn2Branch GitHub repository [https://github.com/ds4dm/learn2branch/tree/master](https://github.com/ds4dm/learn2branch/tree/master), by [Gasse et al., (2020)](https://arxiv.org/abs/1906.01629).

## To get the performance metrics of instance groups (Gurobi):
```bash
python run_gurobi.py --file instances.txt
```
instances.txt is a pickled list of paths to instances.

## Experiments with Learning to Branch:
### To train a model with mixed distributions
```bash
python 03_train_gcnn-modified.py mixed_5_domain -m baseline -s SEED --n_samples_per_domain N
```
### To train seperate models with homogeneous distributions 
```bash
python 03_train_gcnn-modified.py DISTRIBUTION_FOLDER -m baseline -s SEED --n_samples_per_domain N
```


### To evaluate the performance of trained models in MILP solving
```bash
python 05_evaluate.py --model_folder MODEL_FOLDER --ML True --domain TEST_DISTRIBUTION --seed SEED --save_folder SAVE_PATH
```
### To evaluate the performance of SCIP baseline
```bash
python 05_evaluate.py --ML False --domain TEST_DISTRIBUTION --seed SEED --save_folder SAVE_PATH
```

MODEL_FOLDER is mixed_5_domain_n{N} for models trained on mixed distributions and {DISTRIBUTION_FOLDER}_n{N} for models trained on homogeneous distributions. TEST_DISTRIBUTION is the distribution that we test the trained policies with.

We tested with N=80,160,320 and ran with SEED=0,1,2,3,4 for all experiments with both mixed distributions and homogeneous distributions.
