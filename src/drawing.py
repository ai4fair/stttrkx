#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    output_base = os.path.dirname(os.path.abspath(__file__))
except KeyError as e:
    print("Require the directory for outputs")
    print("Given by environment variable: TRKXOUTPUTDIR")
    raise e


# detector file
detector_path = os.path.join(output_base, 'stt.csv')

# TODO: fetch functions from the utils_math.py and remove them from here.

# Polar to Cartesian
def polar_to_cartesian(r, phi):
    """Convert cylinderical to cartesian coordinates. 
    Offset scaling [r*100, phi*np.pi, z*100]"""
    theta = phi * np.pi
    x = r * np.cos(theta)*100
    y = r * np.sin(theta)*100
    return x, y


# Cylinerical to Cartesian
def cylindrical_to_cartesian(r, phi, z):
    """Convert cylinderical to cartesian coordinates. 
    Offset scaling [r*100, phi*np.pi, z*100]"""
    theta = phi * np.pi
    x = r * np.cos(theta)*100
    y = r * np.sin(theta)*100
    z = z * 100
    return x, y, z


def detector_layout(figsize=(10, 10), save_fig=False):
    """Draw Detector (STT) Layout, intended as base plot for further plotting"""
    
    plt.close('all')
    
    # init subplots
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    # detector layout    
    det = pd.read_csv(detector_path)
    nkw = det.query('skewed==0')  # non-skewed
    skw = det.query('skewed==1')  # skewed: both +ve/-ve polarity
    ax.scatter(nkw.x.values, nkw.y.values, s=20, facecolors='none', edgecolors='lightgreen')
    ax.scatter(skw.x.values, skw.y.values, s=20, facecolors='none', edgecolors='coral')
    
    # plotting params
    ax.set_xlabel('x [cm]', fontsize=10)
    ax.set_ylabel('y [cm]', fontsize=10)
    ax.set_xlim(-41, 41)
    ax.set_ylim(-41, 41)
    ax.grid(False)
    
    return fig, ax

# Using Object Oriented API
# -----------------------------------------------------------------------------
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
        # TODO: here z=isochrone radius, one should add a  separate variable for isochrone in feature_data
        ax.scatter(x[idx], y[idx], s=(z[idx]*50), label='particle_id: {}'.format(pid))

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


# Using Object Oriented API
# -----------------------------------------------------------------------------
def draw_event(event=None, figsize=(10, 10), save_fig=False):
    """Draw a single event using 'event' DataFrame."""
    
    # OOP Method #1
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    
    e_ids = np.unique(event.event_id.values)
    p_ids = np.unique(event.particle_id.values)
    det = pd.read_csv(detector_path)
    
    nkw = det.query('skewed==0')  # non-skewed
    skw = det.query('skewed==1')  # skewed: both +ve/-ve polarity
    
    ax.scatter(nkw.x.values, nkw.y.values, s=20, facecolors='none', edgecolors='lightgreen')
    ax.scatter(skw.x.values, skw.y.values, s=20, facecolors='none', edgecolors='coral')

    for i in p_ids:
        df_ = event.loc[event.particle_id == i]
        ax.scatter(df_.x.values, df_.y.values, s=(df_.isochrone*150).values, label='particle_id: {}'.format(i))
    
    ax.set_title('Event ID # {}'.format(e_ids))
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
        fig.savefig('event_{}.png'.format(e_ids))
    return fig


def draw_event_v2(hits=None, tubes=None, particles=None, truth=None, event_id=0, figsize=(10, 10), save_fig=False):
    """Draw a single event using 'hits', 'tubes', 'particles' and 'truth' DataFrames."""
    
    # OOP Method #2
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(nrows=1, ncols=1)
    
    p_ids = np.unique(particles.particle_id.values)
    det = pd.read_csv(detector_path)
    
    nkw = det.query('skewed==0')  # non-skewed
    skw = det.query('skewed==1')  # skewed: both +ve/-ve polarity
    
    ax.scatter(nkw.x.values, nkw.y.values, s=20, facecolors='none', edgecolors='lightgreen')
    ax.scatter(skw.x.values, skw.y.values, s=20, facecolors='none', edgecolors='coral')

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


# Using Pyplot API
# -----------------------------------------------------------------------------
def draw_single_event(event=None, figsize=(10, 10), save_fig=False):
    """Draw a single event using 'event' DataFrame."""
    
    plt.close('all')
    fig = plt.figure(figsize=figsize)
    
    e_ids = np.unique(event.event_id.values)
    p_ids = np.unique(event.particle_id.values)
    det = pd.read_csv(detector_path)
    
    nkw = det.query('skewed==0')  # non-skewed
    skw = det.query('skewed==1')  # skewed: both +ve/-ve polarity
    
    plt.scatter(nkw.x.values, nkw.y.values, s=20, facecolors='none', edgecolors='lightgreen')
    plt.scatter(skw.x.values, skw.y.values, s=20, facecolors='none', edgecolors='coral')

    for i in p_ids:
        df_ = event.loc[event.particle_id == i]
        plt.scatter(df_.x.values, df_.y.values, s=(df_.isochrone*150).values, label='particle_id: {}'.format(i))
    
    plt.title('Event ID # {}'.format(e_ids))
    plt.xlabel('x [cm]', fontsize=10)
    plt.ylabel('y [cm]', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlim((-41, 41))
    plt.ylim((-41, 41))
    plt.grid(False)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    
    if save_fig:
        plt.savefig('event_{}.png'.format(e_ids))
    return fig


def draw_single_event_v2(hits=None, tubes=None, particles=None, truth=None,
                         event_id=0, figsize=(10, 10), save_fig=False):
    """Draw a single event using 'hits', 'tubes', 'particles' and 'truth' DataFrames."""
    
    plt.close('all')
    fig = plt.figure(figsize=figsize)
    
    p_ids = np.unique(particles.particle_id.values)
    det = pd.read_csv(detector_path)
    
    nkw = det.query('skewed==0')  # non-skewed
    skw = det.query('skewed==1')  # skewed: both +ve/-ve polarity
    
    plt.scatter(nkw.x.values, nkw.y.values, s=20, facecolors='none', edgecolors='lightgreen')
    plt.scatter(skw.x.values, skw.y.values, s=20, facecolors='none', edgecolors='coral')

    for i in p_ids:
        df_ = hits.loc[truth.particle_id == i]
        plt.scatter(df_.x.values, df_.y.values, s=(tubes.loc[truth.particle_id == i].isochrone*100).values,
                    label='particle_id: {}'.format(i))
    
    plt.title('Event ID # {}'.format(event_id))
    plt.xlabel('x [cm]', fontsize=10)
    plt.ylabel('y [cm]', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlim((-41, 41))
    plt.ylim((-41, 41))
    plt.grid(False)
    plt.legend(fontsize=10, loc='best')

    plt.tight_layout()
    if save_fig:
        plt.savefig('event_{}.png'.format(event_id))
    return fig
