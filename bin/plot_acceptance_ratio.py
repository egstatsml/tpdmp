#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt

bps_f = '/home/XXXXX/data/XXXXX/data/pdmp/uai2023_acceptance_ratio_1_2/mnist_lenet_bps_interpolation/acceptance_ratio.npy'
cov_f = '/home/XXXXX/data/XXXXX/data/pdmp/uai2023_acceptance_ratio_1_2/mnist_lenet_cov_pbps_interpolation/acceptance_ratio.npy'
boomerang_f = '/home/XXXXX/data/XXXXX/data/pdmp/uai2023_acceptance_ratio_1_2/mnist_lenet_boomerang_interpolation/acceptance_ratio.npy'


bps = np.load(bps_f)
cov = np.load(cov_f)
boomerang = np.load(boomerang_f)

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
# rectangular box plot
flierprops = dict(marker='o', markerfacecolor='black', markersize=6,
                  linestyle='none', markeredgecolor='black', alpha=0.2)
bplot1 = ax1.boxplot([bps, cov, boomerang],
                     # flierprops=flierprops,
                     showfliers=False,
                     labels=['BPS', '$\sigma$BPS', 'Boomerang'],
                     vert=True,  # vertical box alignment
                     patch_artist=True)  # fill with color

# fill with colors
colors = ['#40a02b', '#04a5e5', '#dd7878']
for patch, color in zip(bplot1['boxes'], colors):
  patch.set_facecolor(color)
for median in bplot1['medians']:
    median.set_color('black')
ax1.set_ylim(0.45, 1.3)
plt.savefig('acceptence_box_1_2.pdf')
