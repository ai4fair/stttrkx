#!/usr/bin/env python
# coding: utf-8

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import trackml.dataset
from typing import Any
from collections import namedtuple
from .utils_math import cylindrical_to_cartesian

# TODO: skewed or non-skewed selection: make layer_id column for both cases.

try:
    output_base = os.path.dirname(os.path.abspath(__file__))
except KeyError as e:
    print("Require the directory for outputs")
    print("Given by environment variable: TRKXOUTPUTDIR")
    raise e


# detector file
detector_path = os.path.join(output_base, 'stt.csv')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
Data = namedtuple('Data', ['hits', 'tubes', 'particles', 'truth', 'event', 'event_file'])


# SttTorchDataReader Class
class SttTorchDataReader(object):
    """Torch Geometric Data Reader from an Input Directory."""

    def __init__(self, input_dir: str):
        """Initialize Instance Variables in Constructor"""

        self.path = input_dir
        all_files = sorted(glob.glob(os.path.join(input_dir, "*")))
        self.nevts = len(all_files)
        self.all_evtids = [os.path.basename(x) for x in all_files]

    def read(self, evtid: int = None):
        """Read an Event from the Input Directory."""
        event_fname = os.path.join(self.path, "{}".format(evtid))
        event = torch.load(event_fname, map_location=device)
        return event

    def __call__(self, evtid: int, *args: Any, **kwds: Any) -> Any:
        return self.read(evtid)


# Draw Reader Event::  Using Object-Oriented API
def Draw_TorchReader_Event(feature_data, figsize=(10, 10), save_fig=False):
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


# SttCSVDataReader Class
class SttCSVDataReader(object):
    """Reader for Tracks from GNN Stage (Test Step by GNNBuilder Callback)"""
    
    def __init__(self, path: str, selection: bool, noise: bool, skewed: bool):
        """Initialize Instance Variables in Constructor"""
        self._path = path
        self._noise = noise
        self._skewed = skewed
        self._selection = selection
        self._detector = None

        self._evtid = None
        self._hits = None
        self._particles = None
        self._truth = None
        self._cells = None
        self._event = None

        all_files = sorted(glob.glob(os.path.join(self._path, "*")))
        self.nevts = len(all_files)
        self.all_evtids = [os.path.basename(x) for x in all_files]
    
    def read(self, evtid: int = None):
        """Prepare event similar to event_utils/select_hit() in Processing."""
        
        # create event_prefix from evtid
        prefix = "event{:010d}".format(evtid)
        event_prefix = os.path.join(os.path.expandvars(self._path), prefix)
        
        try:
            all_data = trackml.dataset.load_event(event_prefix)  # return with default order
        except Exception as e:
            return e
            
        if all_data is None:
            return False
        
        hits, cells, particles, truth = all_data
        
        # preprocess particles dataframe e.g. nhits, drop_duplicates, etc.
        particles['nhits'] = particles.groupby(['particle_id'])['nhits'].transform('count')
        particles.drop_duplicates(inplace=True, ignore_index=True)
        
        if self._selection:
            # just keep protons, pions, don't forget resetting index and dropping old one.
            particles = particles[particles['pdgcode'].isin([-2212, 2212, -211, 211])].reset_index(drop=True)
        
        # merge some columns of tubes to the hits, I need isochrone, skewed & sector_id
        hits = hits.merge(cells[["hit_id", "isochrone", "skewed", "sector_id"]], on="hit_id")
    
        # add evtId to hits
        hits = hits.assign(event_id=evtid)
        
        # add pT to particles
        px = particles.px
        py = particles.py
        pz = particles.pz
        
        pt = np.sqrt(px**2 + py**2)
        p = np.sqrt(px**2 + py**2 + pz**2)
        ptheta = np.arccos(pz/p)
        peta = -np.log(np.tan(0.5*ptheta))
        
        particles = particles.assign(pt=pt, peta=peta)
        
        # assign vars
        self._evtid = evtid
        self._hits = hits
        self._particles = particles
        self._truth = truth
        self._cells = cells
        
        # compose event
        self.merge_truth_info_to_hits()
        
        # prepare Data tuple
        data = Data(self._hits, 
                    self._cells, 
                    self._particles, 
                    self._truth, 
                    self._event,  # merged event similar to select_hits()
                    os.path.abspath(event_prefix)
                    )
        
        return data

    def merge_truth_info_to_hits(self):
        """Merge truth information ('truth', 'particles') to 'hits'. 
        Then calculate and add derived variables to the event."""
        
        hits = self._hits
        
        # account for noise
        if self._noise:
            # runs if noise=True
            truth = self._truth.merge(self._particles, on="particle_id", how="left")
        else:
            # runs if noise=False
            truth = self._truth.merge(self._particles, on="particle_id", how="inner")
        
        # skip skewed tubes
        if self._skewed is False:
            hits = hits.query('skewed==0')
            
            # rename layers from 0,1,2...,17 & assign to "layer" column
            vlids = hits.layer_id.unique()
            n_det_layers = len(vlids)
            vlid_groups = hits.groupby(['layer_id'])
            hits = pd.concat([vlid_groups.get_group(vlids[i]).assign(layer=i) for i in range(n_det_layers)])
            self._hits = hits.reset_index(drop=True)
            
        # merge 'hits' with 'truth'
        hits = hits.merge(truth, on="hit_id", how='left')
    
        # add new features to 'hits'
        x = hits.x
        y = hits.y
        z = hits.z
        absz = np.abs(z)
        r = np.sqrt(x**2 + y**2)  # in 2D
        r3 = np.sqrt(r**2 + z**2)  # in 3D
        phi = np.arctan2(hits.y, hits.x)
        theta = np.arccos(z/r3)
        eta = -np.log(np.tan(theta/2.))

        tpx = hits.tpx
        tpy = hits.tpy
        tpt = np.sqrt(tpx**2 + tpy**2)
        
        # add derived quantities to 'hits'
        hits = hits.assign(r=r, phi=phi, eta=eta, r3=r3, absZ=absz, tpt=tpt)
        
        self._event = hits
    
    def __call__(self, evtid: int, *args: Any, **kwds: Any) -> Any:
        return self.read(evtid)


# Draw Reader Event:: Using Object-Oriented API
def Draw_CSVReader_Event(data=None, figsize=(10, 10), save_fig=False):
    """Draw a single event produced by SttCSVReader class."""
    
    # OOP Method #1
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    
    event = data.event
    e_ids = int(data.event_file[-10:])
    p_ids = np.unique(event.particle_id.values)
    det = pd.read_csv(detector_path)
    
    nkw = det.query('skewed==0')  # non-skewed
    skw = det.query('skewed==1')  # skewed: both +ve/-ve polarity
    
    ax.scatter(nkw.x.values, nkw.y.values, s=45, facecolors='none', edgecolors='lightgreen')
    ax.scatter(skw.x.values, skw.y.values, s=45, facecolors='none', edgecolors='coral')

    for i in p_ids:
        df_ = event.loc[event.particle_id == i]
        ax.scatter(df_.x.values, df_.y.values, 
                   # s=(df_.isochrone*3000/45).values,
                   label='particle_id: {}'.format(i))
    
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
