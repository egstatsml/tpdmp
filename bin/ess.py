#!/usr/bin/env python3
import time
from abc import ABCMeta
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import tensorflow_probability as tfp
from tensorflow import keras

import tensorflow as tf
from tbnn.nn.mlp import MLP
from tbnn.nn.conv import Conv
from tbnn.utils import utils, display

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
from glob import glob
import json
from sklearn.metrics import accuracy_score
from tbnn.pdmp.resnet import ResNetLayer
from tbnn.pdmp.frn import FRN


from sklearn.decomposition import PCA

tfd = tfp.distributions

data_models = ['cifar_100_resnet', 'cifar_10_resnet', 'svhn_resnet',
              'mnist_lenet', 'fashion_mnist_lenet']

save_dirs = ['outdir/XXXX_bps_interpolation',
             'outdir/XXXX_cov_pbps_interpolation',
             'outdir/XXXX_boomerang_interpolation',
             'outdir/XXXX_hmc',
             'outdir/XXXX_sgld',
             'outdir/XXXX_sgld_no_decay']


for data_model in data_models:
  print(f'Data model = {data_model}\n')
  for i in range(len(save_dirs)):
    try:
      save_dir = save_dirs[i]
      # replace the XXXX with the data model
      save_dir = save_dir.replace('XXXX', data_model)
      print(f'save dir = {save_dir}')
      chain_files = glob(os.path.join(save_dir, 'chain_*.pkl'))
      idx = 0
      print(chain_files)
      # load in the first chain file to get the dimensions of everything
      with open(chain_files[0], 'rb') as f:
          chain = pickle.load(f)

      samples = np.concatenate([x.numpy().reshape(x.shape[0], -1) for x in chain], axis=1)
      samples = samples.reshape(samples.shape[0], -1)
      pca = PCA(whiten=True).fit(samples)
      transformed = pca.transform(samples)
      ess = tfp.mcmc.effective_sample_size(transformed)
      print(f' ess max = {ess[0]}, ess second = {ess[1]}, ess min {ess[-1]}')
      var = np.concatenate([np.ravel(np.var(x)) for x in chain])
    except Exception as e:
      print(f'error on {save_dir}')
