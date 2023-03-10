#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils_math import polar_to_cartesian, cylindrical_to_cartesian

try:
    output_base = os.path.dirname(os.path.abspath(__file__))
except KeyError as e:
    print("Require the directory for outputs")
    print("Given by environment variable: TRKXOUTPUTDIR")
    raise e


# detector file (STT)
detector_path = os.path.join(output_base, 'stt.csv')


# Draw STT: Using Object Oriented API
def detector_layout(figsize=(10, 10)):
    """Draw Detector (STT) Layout, intended as base for further plotting"""
    
    plt.close('all')
    
    # init subplots
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    # detector layout    
    det = pd.read_csv(detector_path)
    nkw = det.query('skewed==0')  # non-skewed
    skw = det.query('skewed==1')  # skewed: both +ve/-ve polarity
    ax.scatter(nkw.x.values, nkw.y.values, s=44, facecolors='none', edgecolors='lightgreen')
    ax.scatter(skw.x.values, skw.y.values, s=44, facecolors='none', edgecolors='coral')
    
    # plotting params
    ax.set_xlabel('x [cm]', fontsize=15)
    ax.set_ylabel('y [cm]', fontsize=15)
    ax.set_xlim(-41, 41)
    ax.set_ylim(-41, 41)
    ax.grid(False)
    
    return fig, ax


# Draw CSV Event:: Using Object Oriented API
def draw_csv_event(hits=None, tubes=None, particles=None, truth=None, event_id=0, figsize=(10, 10), save_fig=False):
    """Draw a single event using 'hits', 'tubes', 'particles' and 'truth' DataFrames."""
    
    # OOP Method #2
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(nrows=1, ncols=1)
    
    p_ids = np.unique(particles.particle_id.values)
    det = pd.read_csv(detector_path)
    
    nkw = det.query('skewed==0')  # non-skewed
    skw = det.query('skewed==1')  # skewed: both +ve/-ve polarity
    
    ax.scatter(nkw.x.values, nkw.y.values, s=44, facecolors='none', edgecolors='lightgreen')
    ax.scatter(skw.x.values, skw.y.values, s=44, facecolors='none', edgecolors='coral')

    for i in p_ids:
        df_ = hits.loc[truth.particle_id == i]
        ax.scatter(df_.x.values, df_.y.values, s=(tubes.loc[truth.particle_id == i].isochrone*100).values,
                   label='particle_id: {}'.format(i))
    
    ax.set_title('Event ID # {}'.format(event_id))
    ax.set_xlabel('x [cm]', fontsize=10)
    ax.set_ylabel('y [cm]', fontsize=10)
    # ax.set_xticks(xticks, fontsize=10)
    # ax.set_yticks(yticks, fontsize=10)
    ax.set_xlim(-41, 41)
    ax.set_ylim(-41, 41)
    ax.grid(False)
    ax.legend(fontsize=10, loc='best')
    fig.tight_layout()
    
    if save_fig:
        fig.savefig('event_{}.png'.format(event_id))
    return fig


# Draw PyG Event:: Using Object Oriented API
def draw_proc_event(feature_data, figsize=(10, 10), save_fig=False):
    """Draw event from the processing stage, the `feature_data` is pytorch_geometric data."""
    plt.close('all')
    
    # init subplots
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    # detector layout    
    det = pd.read_csv(detector_path)
    nkw = det.query('skewed==0')  # non-skewed
    skw = det.query('skewed==1')  # skewed: both +ve/-ve polarity
    ax.scatter(nkw.x.values, nkw.y.values, s=44, facecolors='none', edgecolors='lightgreen')
    ax.scatter(skw.x.values, skw.y.values, s=44, facecolors='none', edgecolors='coral')
    
    # event Id
    e_id = int(feature_data.event_file[-10:])
        
    # feature data
    x, y, z = cylindrical_to_cartesian(r=feature_data.x[:, 0], phi=feature_data.x[:, 1], z=feature_data.x[:, 2])
    
    # particle tracks
    p_ids = np.unique(feature_data.pid)
    for pid in p_ids:
        idx = feature_data.pid == pid
        # TODO: here z=isochrone radius, one should add a  separate variable for isochrone in feature_data
        ax.scatter(x[idx],
                   y[idx], 
                   # s=(z[idx]*100),
                   label='particle_id: {}'.format(pid))

    # plotting params
    ax.set_title('Event ID # %d' % e_id)
    ax.set_xlabel('x [cm]', fontsize=15)
    ax.set_ylabel('y [cm]', fontsize=15)
    ax.set_xlim(-41, 41)
    ax.set_ylim(-41, 41)
    ax.grid(False)
    ax.legend(fontsize=10, loc='best')
    fig.tight_layout()
    
    if save_fig:
        fig.savefig('event_%d.png' % e_id)
    return fig
