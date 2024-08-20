#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .detector import detector_layout
from .math_utils import polar_to_cartesian

try:
    output_base = os.path.dirname(os.path.abspath(__file__))
except KeyError as e:
    print("Require the directory for outputs")
    print("Given by environment variable: TRKXOUTPUTDIR")
    raise e


# detector file (STT)
detector_path = os.path.join(output_base, 'stt.csv')


# Draw SttCSVDataReader Event:: Using Object-Oriented API
def Visualize_CSVEvent(event=None, figsize=(10, 10), save_fig=False):
    """Draw an event produced by SttCSVDataReader class."""
    
    # draw detector layout
    fig, ax = detector_layout(figsize=(10,10))
    
    # get colormap for consistent track colors
    cmap = ['blue', 'red', 'green', 'orange', 'purple', 'magenta', 'black',
            'gray', 'lime', 'teal', 'navy', 'maroon', 'olive', 'indigo', 'cyan']
    
    # get event_id and particle_id
    evtid = np.unique(event.event_id.values)
    unique_pids =  np.unique(event.particle_id.values)
    
    # draw particles
    for pid in unique_pids:
        df = event.loc[event.particle_id == pid]
        ax.scatter(df.x.values, df.y.values, color=cmap[pid], label=f'particle_id: {pid}')  
    
    # axis params
    ax.legend(fontsize=12, loc='best')
    fig.tight_layout()
    
    # save figure
    if save_fig:
        fig.savefig('event_{}.pdf'.format(evtid))
    
    return fig
 

# Draw SttTorchDataReader Event::  Using Object-Oriented API
def Visualize_TorchEvent(feature_data, figsize=(10, 10), save_fig=False):
    """Draw an event produced by SttTorchDataReader class."""
    
    # draw detector layout
    fig, ax = detector_layout(figsize=(10,10))
    
    # get colormap for consistent track colors
    cmap = ['blue', 'red', 'green', 'orange', 'purple', 'magenta', 'black',
            'gray', 'lime', 'teal', 'navy', 'maroon', 'olive', 'indigo', 'cyan']
    
    # get event_id and particle_id
    evtid = int(feature_data.event_file[-10:])
    unique_pids = np.unique(feature_data.pid)
    
    # feature data
    x, y = polar_to_cartesian(r=feature_data.x[:, 0], phi=feature_data.x[:, 1])
    
    # draw particles
    for pid in unique_pids:
        mask = (feature_data.pid == pid)
        ax.scatter(x[mask], y[mask], color=cmap[pid], label=f'particle_id: {pid}')

    # axis params
    ax.legend(fontsize=12, loc='best')
    fig.tight_layout()
    
    # save figure
    if save_fig:
        fig.savefig('event_{}.pdf'.format(evtid))

    return fig

