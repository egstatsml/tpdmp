#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def old():
  """old method I would use"""
  # load in the data sets
  sgld_dir = '/home/ethan/data/ethan/data/pdmp/uai2023_sgd/cifar_10_resnet_sgld'
  boomerang_dir = '/home/ethan/data/ethan/data/pdmp/uai2023_sgd/cifar_10_resnet_boomerang_interpolation'

  # load in the entropies
  sgld_in_path = os.path.join(sgld_dir, 'cifar_10_entropy.npy')
  sgld_out_path = os.path.join(sgld_dir, 'svhn_entropy.npy')
  boomerang_in_path = os.path.join(boomerang_dir, 'cifar_10_entropy.npy')
  boomerang_out_path = os.path.join(boomerang_dir, 'svhn_entropy.npy')

  sgld_in = np.load(sgld_in_path)
  sgld_out = np.load(sgld_out_path)
  boomerang_in = np.load(boomerang_in_path)
  boomerang_out = np.load(boomerang_out_path)

  sgld_color = '#dd7878'
  boomerang_color = '#7287fd'
  plt.figure()
  plt.hist(sgld_in, 100, density=True, alpha=0.9, color=sgld_color)
  plt.hist(boomerang_in, 100, density=True, alpha=0.8, color=boomerang_color)
  plt.xlabel('entropy')
  plt.savefig('in_entropy.pdf', bbox_inches='tight')
  plt.clf()

  plt.hist(sgld_out, 100, density=True, alpha=0.9, color=sgld_color, label='SGLD')
  plt.hist(boomerang_out, 100, density=True, alpha=0.8, color=boomerang_color, label='Boomerang')
  plt.xlabel('entropy')
  plt.legend()
  plt.savefig('out_entropy.pdf', bbox_inches='tight')


def get_sampler(out_dir):
  if 'boomerang' in out_dir.lower():
    return 'boomerang'
  elif 'sgld_no_decay' in out_dir.lower():
    return 'sgld_no_decay'
  elif 'sgld' in out_dir.lower():
    return 'sgld'
  elif 'cov_pbps' in out_dir.lower():
    return 'cov_pbps'
  elif 'bps' in out_dir.lower():
    return 'bps'
  else:
    raise ValueError('Invalid path stored')


def main(args):
  # load in the entropies
  sampler = get_sampler(args.out_dir)
  in_data =  np.load(os.path.join(args.out_dir, f'{args.in_data}_entropy.npy'))
  out_data =  np.load(os.path.join(args.out_dir, f'{args.out_data}_entropy.npy'))

  in_color = '#7287fd'
  out_color = '#dd7878'
  plt.figure()
  plt.hist(in_data, 100, density=True, alpha=0.9, color=in_color)
  plt.hist(out_data, 100, density=True, alpha=0.8, color=out_color)
  plt.xlabel('entropy')
  plt.savefig(os.path.join(args.out_dir, 'id_ood_entropy.pdf'), bbox_inches='tight')
  plt.savefig(f'../entropy_figs/{args.in_data}_{args.out_data}_{sampler}_id_ood_entropy.pdf', bbox_inches='tight')
  plt.clf()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      prog='test_conv_new',
      epilog=main.__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('in_data', type=str, help='in dist. data')
  parser.add_argument('out_data', type=str, help='OOD data')
  parser.add_argument('--out_dir',
                      type=str,
                      default='./out',
                      help='out directory where data is saved')
  args = parser.parse_args(sys.argv[1:])
  # main(args)
  old()
