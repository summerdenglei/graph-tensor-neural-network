Code for "Graph-Tensor Neural Networks for Network Traffic Data Imputation"

generating_data.m: Abilene dataset (with size 168*12*24*144) contains traffic data within 168 days. We use data within 140 days (with size 140*12*24*144) for training and the rest for testing (with size 28*12*24*144).

Require:
h5py 3.2.1
tensorflow-gpu 1.14.0
tensorflow-estimator 1.14.0
numpy 1.21.0
scipy 1.2.0
