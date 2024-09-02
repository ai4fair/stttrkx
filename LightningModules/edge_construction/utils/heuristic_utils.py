#!/usr/bin/env python
# coding: utf-8

"""Utilities for Input Graph Construction"""

import logging
import scipy as sp
import numpy as np
import pandas as pd
import torch

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Layerwise Heuristic
def select_layerwise_input_edges(lg1, lg2, filtering=True):
    """Select edges using a particular phi range or sectors. Currently, I am selecting edges 
    only in the neighboring sectors i.e. hit1 is paired with hit2 in immediate sectors only."""
    
    # Start with all possible pairs of hits, sector_id is for sectorwise selection
    keys = ['event_id', 'r', 'phi', 'isochrone', 'sector_id']
    hit_pairs = lg1[keys].reset_index().merge(lg2[keys].reset_index(), on='event_id', suffixes=('_1', '_2'))
    
    if filtering:
        dSector = (hit_pairs['sector_id_1'] - hit_pairs['sector_id_2'])
        sector_mask = ((dSector.abs() < 2) | (dSector.abs() == 5))
        edges = hit_pairs[['index_1', 'index_2']][sector_mask]
    else:
        edges = hit_pairs[['index_1', 'index_2']]
        
    return edges


def construct_layerwise_input_edges(hits, layer_pairs, filtering=True):
    """Construct edges between hit pairs in adjacent layers"""
    
    edges = []
    
    # Loop over layer pairs to construct edges
    layer_groups = hits.groupby('layer')
    for (l1, l2) in layer_pairs:
        
        # Find and join all hit pairs
        try:
            lg1 = layer_groups.get_group(l1)
            lg1 = layer_groups.get_group(l2)
        # If an event has no hits on a layer, we get a KeyError.
        # In that case we just skip to the next layer pair
        except KeyError as e:
            logging.info('skipping empty layer: %s' % e)
            continue
        
        # Construct the edges
        edges.append(select_layerwise_input_edges(lg1, lg2, filtering))
    
    return edges


def get_layerwise_input_edges(hits, filtering=True):
    """Build edge_index list for GNN stage."""
    
    # construct layer pairs
    n_layers = hits.layer.unique().shape[0]
    layers = np.arange(n_layers)
    layer_pairs = np.stack([layers[:-1], layers[1:]], axis=1)
    
    # construct edges
    edge_list = construct_layerwise_input_edges(hits, layer_pairs, filtering)
    
    # concatenate edges
    edges = pd.concat(edge_list)
    
    # return all edges as ndarray
    input_edges = edges.to_numpy().T
    return input_edges
 
