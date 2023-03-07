#!/usr/bin/env python
# coding: utf-8
import sys, os, glob, yaml
import math
import random
import numpy as np
import pandas as pd
import trackml.dataset
import seaborn as sns
import torch
import matplotlib.pyplot as plt
from src import Compose_Event
sys.path.append('src')


suffix = '-hits.csv'
input_dir='./data_sets/ctd2022/fwp_train_100k'
all_files = os.listdir(input_dir)
file_prefixes = sorted(os.path.join(input_dir, f.replace(suffix, ''))
                       for f in all_files if f.endswith(suffix))

i = 0
pt_l = []
nhits_l = []

print("     Total Files:", len(file_prefixes))
for f in range(len(file_prefixes[50])):
    i = i+1
    if i != 0 and i%1000 == 0:
        print("Processed Events:", i)
    
    event = Compose_Event(file_prefixes[f], selection=False, noise=False, skewed=False)
        
    # get nhits and pt
    nhits = event.nhits.values
    pt = event.pt.values

    pt_l.append(pt)
    nhits_l.append(nhits)

# print("# of nhits: ", sum(nhits_l))
# print("# of pt   : ", sum(pt_l))


nhits = np.asarray(nhits_l)
pt = np.asarray(pt_l)



# Plot Histograms
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(1,2, sharey=True)

nbins=100


ax[0].hist(nhits, bins=nbins, edgecolor='None', alpha=0.8, label='Nhits', color='blue')
ax[1].hist(pt, bins=nbins, edgecolor='None', alpha=0.8, label='pT', color='orange')

# plotting params
# ax.set_title('Edge Construction')
ax[0].set_xlabel('Nhits', fontsize=20)
ax[1].set_xlabel('pT', fontsize=20)

ax[0].set_ylabel('Counts', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.tick_params(axis='both', which='minor', labelsize=15)
# ax.set_xscale('log')
ax[0].set_yscale('log')
# ax.set_xlim(-41, 41)
# ax.set_ylim(-41, 41)
ax.grid(False)
ax.legend(fontsize=20, loc='best')
fig.tight_layout()
fig.savefig("hist.pdf")
fig.show()


