#!/usr/bin/env python
# coding: utf-8

"""Drawing functions for the torch-geometric data produced by the Processing stage."""

import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


try:
    cwd = os.path.dirname(os.path.abspath(__file__))
    base_dir = pathlib.Path(cwd).parents[2]
except KeyError as e:
    print("Require the directory for outputs")
    print("Given by environment variable: TRKXOUTPUTDIR")
    raise e


# detector file
detector_path = os.path.join(base_dir, 'src', 'stt.csv')


# Cylinerical to Cartesian
def cylindrical_to_cartesian(r, phi, z):
    """Convert cylinderical to cartesian coordinates. 
    Offset scaling [r*100, phi*np.pi, z*100]"""
    theta = phi * np.pi
    x = r * np.cos(theta)*100
    y = r * np.sin(theta)*100
    z = z * 100
    return x, y, z


# Using Object Oriented API
def draw_proc_event(feature_data, figsize=(10, 10), save_fig=False):
    """Draw event from the processing stage, the `feature_data` is pytorch_geometric data."""
    plt.close('all')
    
    # init subplots
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    # detector layout    
    det = pd.read_csv(detector_path)
    nkw = det.query('skewed==0')  # non-skewed
    skw = det.query('skewed==1')  # skewed: both +ve/-ve polarity
    ax.scatter(nkw.x.values, nkw.y.values, s=20, facecolors='none', edgecolors='lightgreen')
    ax.scatter(skw.x.values, skw.y.values, s=20, facecolors='none', edgecolors='coral')
    
    # event Id
    e_id = int(feature_data.event_file[-10:])
        
    # feature data
    x, y, z = cylindrical_to_cartesian(r=feature_data.x[:, 0], phi=feature_data.x[:, 1], z=feature_data.x[:, 2])
    
    # particle tracks
    p_ids = np.unique(feature_data.pid)
    for pid in p_ids:
        idx = feature_data.pid == pid
        ax.scatter(x[idx], y[idx], label='particle_id: {}'.format(pid))

    # plotting params
    ax.set_title('Event ID # %d' % e_id)
    ax.set_xlabel('x [cm]', fontsize=10)
    ax.set_ylabel('y [cm]', fontsize=10)
    ax.set_xlim(-41, 41)
    ax.set_ylim(-41, 41)
    ax.grid(False)
    ax.legend(fontsize=10, loc='best')
    fig.tight_layout()
    
    if save_fig:
        fig.savefig('event_%d.png' % e_id)
    return fig
