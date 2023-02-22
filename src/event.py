#!/usr/bin/env python
# coding: utf-8

"""STT Dataset (equivalent to TrackML Dataset)"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import trackml.dataset


try:
    output_base = os.path.dirname(os.path.abspath(__file__))
except KeyError as e:
    print("Require the directory for outputs")
    print("Given by environment variable: TRKXOUTPUTDIR")
    raise e


# detector file
detector_path = os.path.join(output_base, 'stt.csv')


# TODO: Adapt Event class to STT Dataset (from xju2-gnn/heptrkx/dataset/event.py)

# Event Class
class Event(object):
    """An object saving Event info, including hits, particles, truth and cell info"""

    def __init__(self, input_dir: str, noise: bool, skewed: bool):
        """Initialize Instance Variables in Constructor"""
        self._path = input_dir
        self._noise = noise
        self._skewed = skewed
        self._detector = None

        self._evtid = None
        self._hits = None
        self._particles = None
        self._truth = None
        self._cells = None
        self._event = None

    def read(self, evtid: int = None):
        """Read a Single Event Using an Event ID."""

        prefix = "event{:010d}".format(evtid)
        event_prefix = os.path.join(os.path.expandvars(self._path), prefix)

        try:
            all_data = trackml.dataset.load_event(event_prefix, parts=['hits', 'particles', 'truth', 'cells'])
        except Exception as e:
            return e

        if all_data is None:
            return False

        hits, particles, truth, cells = all_data
        hits = hits.assign(event_id=evtid)

        # add pT to particles
        px = particles.px
        py = particles.py
        pz = particles.pz
        pt = np.sqrt(px ** 2 + py ** 2)
        momentum = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
        ptheta = np.arccos(pz / momentum)
        peta = -np.log(np.tan(0.5 * ptheta))
        particles = particles.assign(pt=pt, peta=peta)

        # assign vars
        self._evtid = evtid
        self._hits = hits
        self._particles = particles
        self._truth = truth
        self._cells = cells

        # compose event
        self.merge_truth_info_to_hits()
        return True

    @property
    def particles(self):
        return self._particles

    @property
    def hits(self):
        return self._hits

    @property
    def cells(self):
        return self._cells

    @property
    def truth(self):
        return self._truth

    @property
    def evtid(self):
        return self._evtid

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
            n_det_layers = hits.layer_id.unique().shape[0]
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
        r = np.sqrt(x ** 2 + y ** 2)  # in 2D
        r3 = np.sqrt(r ** 2 + z ** 2)  # in 3D
        phi = np.arctan2(hits.y, hits.x)
        theta = np.arccos(z / r3)
        eta = -np.log(np.tan(theta / 2.))

        tpx = hits.tpx
        tpy = hits.tpy
        tpt = np.sqrt(tpx ** 2 + tpy ** 2)

        # add derived quantities to 'hits'
        hits = hits.assign(r=r, phi=phi, eta=eta, r3=r3, absZ=absz, tpt=tpt)
        self._event = hits

    def reconstructable_pids(self, min_hits=4):
        """Find reconstructable particles with min_hits > 4"""
        truth_particles = self.particles.merge(self.truth, on='particle_id', how='left')
        reconstructable_particles = truth_particles[truth_particles.nhits > min_hits]
        return np.unique(reconstructable_particles.particle_id)

    def filter_hits(self, layers, inplace=True):
        """Keep hits that are in the layers"""
        barrel_hits = self._hits[self._hits.layer.isin(layers)]
        if inplace:
            self._hits = barrel_hits
        return barrel_hits

    def remove_noise_hits(self, inplace=True):
        """Remove noise"""
        no_noise = self._hits[self._hits.particle_id > 0]
        if inplace:
            self._hits = no_noise
        return no_noise

    def remove_duplicated_hits(self, inplace=True):
        """Remove duplicate hits"""
        hits = self._hits.loc[
            self._hits.groupby(['particle_id', 'layer'], as_index=False).r.idxmin()
        ]
        if inplace:
            self._hits = hits
            return self._hits
        else:
            return hits

    def select_hits(self, no_noise, eta_cut):
        """Select hits after applying an eta cut"""
        if no_noise:
            self.remove_noise_hits()

        self._hits = self._hits[np.abs(self._hits.eta) < eta_cut]
        return self._hits

    def count_duplicated_hits(self):
        """Count duplicate hits"""
        # sel is the number of "extra" hits
        # if not duplication, sel = 0; otherwise it is the number of duplication
        sel = self._hits.groupby("particle_id")['layer'].apply(
            lambda x: len(x) - np.unique(x).shape[0]
        ).values
        return sel


# Compose DataFrames into Single Event DataFrame.
def Compose_Event(event_prefix="", noise=False, skewed=True):
    """Merge truth information ('truth', 'particles') to 'hits'.
    Then calculate and add derived variables to the event. Keep
    the necessary columns in the final dataframe."""

    # load data using event_prefix (e.g. path/to/event0000000001)
    hits, tubes, particles, truth = trackml.dataset.load_event(event_prefix)

    # first merge truth & particles on particle_id, assuming
    if noise:
        # runs if noise=True
        truth = truth.merge(
            particles[["particle_id", "vx", "vy", "vz"]], on="particle_id", how="left"
        )
    else:
        # runs if noise=False
        truth = truth.merge(
            particles[["particle_id", "vx", "vy", "vz"]], on="particle_id", how="inner"
        )

    # assign pt (from tpx & tpy ???) and add to truth
    truth = truth.assign(pt=np.sqrt(truth.tpx ** 2 + truth.tpy ** 2))

    # merge some columns of tubes to the hits, I need isochrone, skewed & sector_id
    hits = hits.merge(tubes[["hit_id", "isochrone", "skewed", "sector_id"]], on="hit_id")

    # skip skewed tubes
    if skewed is False:
        hits = hits.query('skewed==0')

        # rename layer_ids from 0,1,2...,17 & assign a new colmn named "layer"
        vlids = hits.layer_id.unique()
        n_det_layers = hits.layer_id.unique().shape[0]
        vlid_groups = hits.groupby(['layer_id'])
        hits = pd.concat([vlid_groups.get_group(vlids[i]).assign(layer=i) for i in range(n_det_layers)])

    # merge hits with truth, but first find r & phi
    hits = hits.assign(r=np.sqrt(hits.x ** 2 + hits.y ** 2), phi=np.arctan2(hits.y, hits.x)).merge(truth, on="hit_id")

    # assign event_id to this event
    event = hits.assign(event_id=int(event_prefix[-10:]))

    return event


# Draw Event:: (Using Object Oriented API)
def Draw_Compose_Event(event=None, figsize=(10,10), save_fig=False):
    """Draw a single event using 'event' DataFrame."""
    
    # OOP Method #1
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    
    e_ids = np.unique(event.event_id.values)[0]
    p_ids = np.unique(event.particle_id.values)
    det = pd.read_csv(detector_path)
    
    nkw = det.query('skewed==0')  # non-skewed
    skw = det.query('skewed==1')  # skewed: both +ve/-ve polarity
    
    ax.scatter(nkw.x.values, nkw.y.values, s=45, facecolors='none', edgecolors='lightgreen')
    ax.scatter(skw.x.values, skw.y.values, s=45, facecolors='none', edgecolors='coral')

    for i in p_ids:
        df_ = event.loc[event.particle_id == i]
        ax.scatter(df_.x.values, df_.y.values,
                   #s=(df_.isochrone*3000/45).values,
                   label='particle_id: {}'.format(i))
    
    ax.set_title('Event ID # {}'.format(e_ids))
    ax.set_xlabel('x [cm]', fontsize=15)
    ax.set_ylabel('y [cm]', fontsize=15)
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


# Draw_Event:: (Using Pyplot API)
def Draw_Single_Event(event=None, figsize=(10, 10), save_fig=False):
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
        plt.scatter(df_.x.values, df_.y.values,
                    # s=(df_.isochrone*150).values, 
                    label='particle_id: {}'.format(i))
    
    plt.title('Event ID # {}'.format(e_ids))
    plt.xlabel('x [cm]', fontsize=15)
    plt.ylabel('y [cm]', fontsize=15)
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
