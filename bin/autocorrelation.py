#!/usr/bin/env python3
import time
from abc import ABCMeta
import pickle
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

import numpy as np
import tensorflow_probability as tfp

import tensorflow as tf
from sklearn.decomposition import PCA
import arviz

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
from glob import glob
import json
tfd = tfp.distributions

# save_dirs = ['/home/XXXXX/data/XXXXX/data/pdmp/uai2023/fashion_mnist_lenet_bps_interpolation',
#              '/home/XXXXX/data/XXXXX/data/pdmp/uai2023/fashion_mnist_lenet_cov_pbps_interpolation',
#              '/home/XXXXX/data/XXXXX/data/pdmp/uai2023/fashion_mnist_lenet_boomerang_interpolation',
#              '/home/XXXXX/data/XXXXX/data/pdmp/uai2023_sgd/fashion_mnist_lenet_sgld',
#              '/home/XXXXX/data/XXXXX/data/pdmp/uai2023_sgd/fashion_mnist_lenet_sgld_no_decay']


save_dirs = ['/home/XXXXX/data/XXXXX/data/pdmp/uai2023_sgd/svhn_resnet_bps_interpolation',
             '/home/XXXXX/data/XXXXX/data/pdmp/uai2023_sgd/svhn_resnet_cov_pbps_interpolation',
             '/home/XXXXX/data/XXXXX/data/pdmp/uai2023_sgd/svhn_resnet_boomerang_interpolation',
             '/home/XXXXX/data/XXXXX/data/pdmp/uai2023_sgd/svhn_resnet_sgld',
             '/home/XXXXX/data/XXXXX/data/pdmp/uai2023_sgd/svhn_resnet_sgld_no_decay',]
sampler_names = ['BPS' ,r'$\sigma$BPS', 'Boomerang', 'SGLD', 'SGLD-ND']
# save_dir = '/home/XXXXX/data/XXXXX/data/pdmp/uai2023/fashion_mnist_lenet_boomerang_interpolation'
# save_dir = '/home/XXXXX/data/XXXXX/data/pdmp/uai2023/fashion_mnist_lenet_bps_interpolation'
# save_dir = '/home/XXXXX/data/XXXXX/data/pdmp/uai2023/fashion_mnist_lenet_cov_pbps_interpolation'
# save_dir = '/home/XXXXX/data/XXXXX/data/pdmp/uai2023/fashion_mnist_lenet_sgld'
idx = 0
samples_list = []
autocorr_list = []
transformed_list = []
for i, dir in enumerate(save_dirs):
  chain_files = glob(os.path.join(dir, 'chain_*.pkl'))
  with open(chain_files[0], 'rb') as f:
    chain = pickle.load(f)
    # get just the weights from the last layer
    samples = np.concatenate([x.numpy().reshape(x.shape[0], -1) for x in chain], axis=1)
    print(samples.shape)
    # samples = chain[0].numpy()
    # print(samples[:, 0])
    # print(samples.reshape(200, -1)[:, 0])
    samples = samples.reshape(samples.shape[0], -1)
    samples_list.append(samples)
    # time.sleep(2)
    pca = PCA(whiten=True).fit(samples)
    # print(samples.shape)
    # print(samples.reshape(samples.shape[0], -1).shape)
    transformed = pca.transform(samples)
    ess = tfp.mcmc.effective_sample_size(transformed)
    print(f'sampler = {sampler_names[i]} ess max = {ess[0]}, ess second = {ess[1]}, ess min {ess[-1]}')
    transformed_list.append(transformed)
    autocorr = arviz.autocorr(np.transpose(transformed))
    print(np.var(transformed[:, 0]))
    # autocorr = arviz.autocorr(np.transpose(samples.reshape(samples.shape[0], -1)))
    print(autocorr.shape)
    autocorr_list.append(autocorr)


plt.figure()
for i in range(0, len(sampler_names)):
  plt.plot(np.squeeze(autocorr_list[i][0, :]), label=sampler_names[i])
  # plt.plot(np.squeeze(samples_list[i][:, 10]), label=sampler_names[i])
plt.xlim([0, 200])
plt.xlabel('lag')
# plt.legend()
plt.savefig('./autocorr_most.pdf',
            bbox_inches='tight')
plt.clf()
for i in range(0, len(sampler_names)):
  plt.plot(np.squeeze(autocorr_list[i][1, :]), label=sampler_names[i])
  # plt.plot(np.squeeze(samples_list[i][:, 10]), label=sampler_names[i])
plt.xlim([0, 200])
plt.xlabel('lag')
# plt.legend()
plt.savefig('./autocorr_second.pdf',
            bbox_inches='tight')
plt.clf()

for i in range(0, len(sampler_names)):
  plt.plot(np.squeeze(autocorr_list[i][-1, :]), label=sampler_names[i])
plt.xlim([0, 200])
plt.xlabel('lag')
# plt.legend()
# plt.legend()
plt.savefig('./autocorr_min.pdf',
            bbox_inches='tight')
plt.clf()
for i in range(0, len(sampler_names)):
  plt.plot(np.squeeze(transformed_list[i][:, 0]), label=sampler_names[i])
plt.xlim([0, 200])
# plt.legend()
plt.savefig('./trace_most.pdf',
            bbox_inches='tight')
plt.clf()
for i in range(0, len(sampler_names)):
  plt.plot(np.squeeze(transformed_list[i][:, 1]), label=sampler_names[i])
plt.xlim([0, 200])
# plt.legend()
plt.savefig('./trace_second.pdf',
            bbox_inches='tight')
plt.clf()

for i in range(0, len(sampler_names)):
  plt.plot(np.squeeze(transformed_list[i][:, -1]), label=sampler_names[i])
plt.xlim([0, 200])
# plt.legend()
plt.savefig('./trace_min.pdf',
            bbox_inches='tight')

plt.clf()
for i in range(0, len(sampler_names)):
  # if sampler_names[i] == 'Boomerang' or sampler_names[i] == 'SGLD':
  #   c = 'green' if sampler_names[i] == 'Boomerang' else 'red'
  plt.plot(np.squeeze(samples_list[i][:,0]), label=sampler_names[i])#, color=c)
plt.xlim([0, 200])
# plt.legend()
plt.savefig('./trace_param.pdf',
            bbox_inches='tight')

# # load in the first chain file to get the dimensions of everything
# with open(chain_files[0], 'rb') as f:
#     chain = pickle.load(f)
# # keep just the last dimension set of layer weights
# samples = chain[-1]
# print(samples[:, 0])
# time.sleep(10)
# samples = samples.numpy().reshape(samples.shape[0], -1)
# print(samples[:, 0])
# pca = PCA(n_components=10).fit(samples)
# m = pca.transform(samples)
# # print(samples)
# print(samples.shape)
# print(m.shape)
# m = np.transpose(m)
# auto = arviz.autocorr(m, axis=1)
# print(auto.shape)
# plt.figure()
# plt.plot(np.squeeze(auto[-1, :]))
# plt.savefig('./autocorr.png')

# print(m[-1, :])

# print(pca.explained_variance_ratio_)
# # print(m[-1, :])
