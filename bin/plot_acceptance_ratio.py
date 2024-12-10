#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})


bps_f = '/home/ethan/data/ethan/data/pdmp/uai2023_acceptance_ratio_1_2/mnist_lenet_bps_interpolation/acceptance_ratio.npy'
cov_f = '/home/ethan/data/ethan/data/pdmp/uai2023_acceptance_ratio_1_2/mnist_lenet_cov_pbps_interpolation/acceptance_ratio.npy'
boomerang_f = '/home/ethan/data/ethan/data/pdmp/uai2023_acceptance_ratio_1_2/mnist_lenet_boomerang_interpolation/acceptance_ratio.npy'
# bps_f = '/home/ethan/data/ethan/data/pdmp/uai2023_sgd/mnist_lenet_bps_interpolation/acceptance_ratio.npy'
# cov_f = '/home/ethan/data/ethan/data/pdmp/uai2023_sgd/mnist_lenet_cov_pbps_interpolation/acceptance_ratio.npy'
# boomerang_f = '/home/ethan/data/ethan/data/pdmp/uai2023_sgd/mnist_lenet_boomerang_interpolation/acceptance_ratio.npy'

# boomerang_f = '/home/ethan/data/ethan/data/pdmp/uai2023_sgd/fashion_mnist_lenet_boomerang_interpolation/acceptance_ratio.npy'

bps = np.load(bps_f)
cov = np.load(cov_f)
boomerang = np.load(boomerang_f)

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
# rectangular box plot
flierprops = dict(marker='o', markerfacecolor='black', markersize=6,
                  linestyle='none', markeredgecolor='black', alpha=0.2)
bplot1 = ax1.boxplot([bps, cov, boomerang],
                     # flierprops=flierprops,
                     showfliers=True,
                     vert=True,  # vertical box alignment
                     patch_artist=True)  # fill with color



labels=['BPS', '$\sigma$BPS', 'Boomerang']
for tick in ax1.yaxis.get_major_ticks():
    tick.label.set_fontsize(14)
ax1.set_xticklabels(labels,
                    rotation=45, fontsize=14)

# num_boxes = 3
# pos = np.arange(num_boxes) + 1
# for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
#     k = tick % 2
#     ax1.text(pos[tick], .95, upper_labels[tick],
#              transform=ax1.get_xaxis_transform(),
#              horizontalalignment='center', size='large')
#              # weight=weights[k], color=box_colors[k])

# fill with colors
colors = ['#40a02b', '#04a5e5', '#dd7878']
for patch, color in zip(bplot1['boxes'], colors):
  patch.set_facecolor(color)
for median in bplot1['medians']:
    median.set_color('black')
ax1.set_ylim(0.45, 2.0)
plt.savefig('acceptence_box_1_2.pdf', bbox_inches='tight')
