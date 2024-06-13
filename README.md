# Distributional_MIPLIB_eval

Evaluation code used in experiments on Distributional MIPLIB

## To get the performance metrics:
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
We tested with N=80,160,320 and ran with SEED=0,1,2,3,4 for all experiments with both mixed distributions and homogeneous distributions.
## Citations

Learning to Branch paper
```
@article{gasse2019exact,
  title={Exact combinatorial optimization with graph convolutional neural networks},
  author={Gasse, Maxime and Ch{\'e}telat, Didier and Ferroni, Nicola and Charlin, Laurent and Lodi, Andrea},
  journal={Advances in neural information processing systems},
  volume={32},
  year={2019}
}
```
