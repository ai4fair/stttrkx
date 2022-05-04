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


# Using Object Oriented API
def draw_event(event=None, figsize=(10, 10), save_fig=False):
    """Draw a Single Event using event Dataframe"""
    
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
    """Draw a Single Event Using hits, tubes, particles, truth Dataframes"""
    
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
def draw_single_event(event=None, figsize=(10, 10), save_fig=False):
    """Draw a Single Event using event Dataframe"""
    
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
    """Draw a Single Event Using hits, tubes, particles, truth Dataframes"""
    
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
