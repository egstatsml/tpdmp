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
from tensorflow.keras.layers import Dense, Conv2D,  MaxPool2D, Flatten, GlobalAveragePooling2D,  BatchNormalization, Layer, Add
from tbnn.pdmp.bps import BPSKernel, IterBPSKernel, CovPBPSKernel, IterCovPBPSKernel, PBPSKernel, IterPBPSKernel, BoomerangKernel, BoomerangIterKernel
from tbnn.pdmp.poisson_process import SBPSampler, PSBPSampler, AdaptivePSBPSampler, AdaptiveSBPSampler

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

tfd = tfp.distributions

save_dir = '/home/XXXXX/data/XXXXX/XXXXX_22/pdmp/mnist_lenet_bps'
chain_files = glob(os.path.join(save_dir, 'chain_*.pkl'))
idx = 0
print(chain_files)
# load in the first chain file to get the dimensions of everything
with open(chain_files[0], 'rb') as f:
    chain = pickle.load(f)
len_params = len(chain)
samples = [[] for i in range(0, len_params)]
for chain_file in chain_files:
    with open(chain_file, 'rb') as f:
        print(chain_file)
        chain = pickle.load(f)
        print(chain[0].shape[0])
    for i in range(0, len_params):
        # only want the i'th index at the moment
        samples[i].append(np.array(chain[i]))

# now want to concatenate them all into a single array
samples = [np.concatenate(samples[i], axis=0) for i in range(0, len_params)]
# now find the ess and put it in one big array
ess = np.concatenate([np.ravel(tfp.mcmc.effective_sample_size(x)) for x in samples])
print(np.min(ess))
print(np.max(ess))
print(np.mean(ess))
