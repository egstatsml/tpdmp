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

for data in ['cifar_10', 'svhn', 'fashion_mnist', 'mnist']:
  if data in ['cifar_10', 'svhn', 'cifar_100']:
    data_model = data + "_resnet"
  else:
    data_model = data + "_lenet"

  # include the legend in the plots only for CIFAR-10
  include_legend = 'cifar_10' in data
  # include_legend = True

  save_dirs = [f'outdir/{data_model}_bps_interpolation',
               f'outdir/{data_model}_cov_pbps_interpolation',
               f'outdir/{data_model}_boomerang_interpolation',
               f'outdir/{data_model}_sgld',
               f'outdir/{data_model}_sgld_no_decay',
               f'outdir/{data_model}_hmc',]
  sampler_names = ['BPS' ,r'$\sigma$BPS', 'Boomerang', 'SGLD', 'SGLD-ND', "SGHMC"]
  idx = 0
  samples_list = []
  autocorr_list = []
  transformed_list = []
  for i, dir in enumerate(save_dirs):
    chain_files = glob(os.path.join(dir, 'chain_*.pkl'))
    print(dir)
    print(chain_files)
    with open(chain_files[0], 'rb') as f:
      chain = pickle.load(f)
      samples = np.concatenate([x.numpy().reshape(x.shape[0], -1) for x in chain], axis=1)
      samples = samples.reshape(samples.shape[0], -1)
      # save to a list as will be using later on for plotting
      samples_list.append(samples)
      pca = PCA(whiten=False).fit(samples)
      transformed = pca.transform(samples)
      print(transformed.shape)
      # to compute ESS over all channels, need as [Samples, features]
      # which is what is returned from PCA
      ess = tfp.mcmc.effective_sample_size(transformed)
      print(f'sampler = {sampler_names[i]} ess max = {ess[0]}, ess second = {ess[1]}, ess min {ess[-1]}')
      ess_arviz_ind = arviz.ess(transformed[:, 0])
      print(f'arviz ind sampler = {sampler_names[i]} ess max = {ess_arviz_ind}')
      ess_tfp_ind = tfp.mcmc.effective_sample_size(transformed[:, 0])
      print(f'tfp individual sampler = {sampler_names[i]} ess max = {ess_tfp_ind}')
      # save to a list as will be using later on for plotting
      transformed_list.append(transformed)
      autocorr = arviz.autocorr(np.transpose(transformed))
      print(f'var first principal component = {np.var(transformed[:, 0])}')
      autocorr_list.append(autocorr)

  plt.figure()
  for i in range(0, len(sampler_names)):
    plt.plot(np.squeeze(autocorr_list[i][0, :]), label=sampler_names[i])
    # plt.plot(np.squeeze(samples_list[i][:, 10]), label=sampler_names[i])
  plt.xlim([0, 200])
  plt.xlabel('lag')
  if include_legend:
    plt.legend()
  plt.savefig(os.path.join('autocorr_figs', f'{data}/autocorr_most.pdf'),
              bbox_inches='tight')
  plt.clf()
  for i in range(0, len(sampler_names)):
    plt.plot(np.squeeze(autocorr_list[i][1, :]), label=sampler_names[i])
    # plt.plot(np.squeeze(samples_list[i][:, 10]), label=sampler_names[i])
  plt.xlim([0, 200])
  plt.xlabel('lag')
  if include_legend:
    plt.legend()
  # plt.legend()
  plt.savefig(os.path.join('autocorr_figs', f'{data}/autocorr_second.pdf'),
              bbox_inches='tight')
  plt.clf()

  for i in range(0, len(sampler_names)):
    plt.plot(np.squeeze(autocorr_list[i][-1, :]), label=sampler_names[i])
  plt.xlim([0, 200])
  plt.xlabel('lag')
  if include_legend:
    plt.legend()

  plt.savefig(os.path.join('autocorr_figs', f'{data}/autocorr_min.pdf'),
              bbox_inches='tight')
  plt.clf()

  for i in range(0, len(sampler_names)):
    plt.plot(np.squeeze(transformed_list[i][:, 0]), label=sampler_names[i])
  plt.xlim([0, 200])
  if include_legend:
    plt.legend()

  plt.xlabel('lag')
  plt.savefig(os.path.join('autocorr_figs', f'{data}/trace_most.pdf'),
              bbox_inches='tight')
  plt.clf()

  for i in range(0, len(sampler_names)):
    plt.plot(np.squeeze(transformed_list[i][:, 1]), label=sampler_names[i])
  plt.xlim([0, 200])
  plt.xlabel('lag')
  if include_legend:
    plt.legend()
  plt.savefig(os.path.join('autocorr_figs', f'{data}/trace_second.pdf'),
              bbox_inches='tight')
  plt.clf()

  for i in range(0, len(sampler_names)):
    plt.plot(np.squeeze(transformed_list[i][:, -1]), label=sampler_names[i])
  plt.xlim([0, 200])
  plt.xlabel('lag')
  if include_legend:
    plt.legend()

  # plt.legend()
  plt.savefig(os.path.join('autocorr_figs', f'{data}/trace_min.pdf'),
              bbox_inches='tight')
  plt.clf()
  for i in range(0, len(sampler_names)):
    # if sampler_names[i] == 'Boomerang' or sampler_names[i] == 'SGLD':
    #   c = 'green' if sampler_names[i] == 'Boomerang' else 'red'
    plt.plot(np.squeeze(samples_list[i][:,-20]), label=sampler_names[i])#, color=c)
  plt.xlim([0, 200])
  plt.xlabel('lag')
  if include_legend:
    plt.legend()
  plt.legend()
  plt.savefig(os.path.join('autocorr_figs', f'{data}/trace_param_last.pdf'),
              bbox_inches='tight')

  plt.clf()
  for i in range(0, len(sampler_names)):
    # if sampler_names[i] == 'Boomerang' or sampler_names[i] == 'SGLD':
    #   c = 'green' if sampler_names[i] == 'Boomerang' else 'red'
    plt.plot(np.squeeze(samples_list[i][:, 20]), label=sampler_names[i])#, color=c)
  plt.xlim([0, 200])
  plt.xlabel('lag')
  if include_legend:
    plt.legend()
  plt.legend()
  plt.savefig(os.path.join('autocorr_figs', f'{data}/trace_param_first.pdf'),
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
