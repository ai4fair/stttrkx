#!/usr/bin/env python
# coding: utf-8

"""Heuristic Methods for Graph Construction."""

import logging
import numpy as np
import pandas as pd
import torch

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Layerwise Heuristic without Samelayer Edges
def construct_layerwise_graph(hits, filtering=True):
    """
    Construct input graph (edges) in adjacent layers and sectors
    
    Parameters
    ----------
    hits : pd.DataFrame
        hits information
    filtering : bool, optional
        if True, require adjacent layers to have a sector difference of 1 or 5
        if False, do not require any sector constraint
    """
    
    # Handle NaN and Null Values
    hits = hits[
        ((~hits.particle_id.isna()) & (hits.particle_id != 0)) & (~hits.vx.isna())
    ]
    hits = hits.drop_duplicates(
        subset=["particle_id", "volume_id", "layer_id", "module_id"]
    )

    # Handle Indexing (Keep order of occurrence)
    hits = hits.reset_index()

    # Rename 'index' column to 'unsorted_index'
    hits = hits.rename(columns={"index": "unsorted_index"}).reset_index(drop=False)

    # Handle Particle_id 0
    hits.loc[hits["particle_id"] == 0, "particle_id"] = np.nan

    # construct layer pairs
    n_layers = hits.layer.unique().shape[0]
    layers = np.arange(n_layers)
    layer_pairs = np.stack([layers[:-1], layers[1:]], axis=1)
    
    # layerwise construction
    input_edges = []
    layer_groups = hits.groupby('layer')
    for (l1, l2) in layer_pairs:
        
        # get l1, l2 layer groups
        try:
            lg1 = layer_groups.get_group(l1)
            lg2 = layer_groups.get_group(l2)
        # If an event has no hits on a layer, we get a KeyError.
        # In that case we just skip to the next layer pair.
        except KeyError as e:
            logging.info('skipping empty layer: %s' % e)
            continue
        
        # get all possible pairs of hits
        keys = ['event_id', 'r', 'phi', 'isochrone', 'sector_id']
        hit_pairs = lg1[keys].reset_index().merge(lg2[keys].reset_index(), on='event_id', suffixes=('_1', '_2'))
        
        # construct edges with/without sector constraint
        if filtering:
            dSector = (hit_pairs['sector_id_1'] - hit_pairs['sector_id_2'])
            sector_mask = ((dSector.abs() < 2) | (dSector.abs() == 5))
            segments = hit_pairs[['index_1', 'index_2']][sector_mask]
        else:
            segments = hit_pairs[['index_1', 'index_2']]
        
        # append edge list
        input_edges.append(segments)

    return input_edges


# Layerwise Heuristic with Samelayer Edges
def extend_layerwise_graph_with_inner_edges (hits, filtering=True):
    """
    This function first constructs the layerwise graph (edges between adjacent layers)
    and then adds the samelayer edges (edges within a layer).

    Parameters
    ----------
    hits : pd.DataFrame
        hits information
    filtering : bool, optional
        if True, require adjacent layers to have a sector difference of 1 or 5
        if False, do not require any sector constraint
    """
    # construct layerwise graph
    input_edges = construct_layerwise_graph(hits, filtering)

    # construct samelayer edges
    layer_groups = hits.groupby('layer')
    for _, lg in layer_groups:
        if len(lg) > 1:

            # same keys as above
            keys = ['event_id', 'r', 'phi', 'isochrone', 'sector_id']
            pairs = lg[keys].reset_index().merge(lg[keys].reset_index(), on='event_id', suffixes=('_1', '_2'))
                    
            # Remove self-loops
            # pairs = pairs[pairs['index_1'] != pairs['index_2']] # bidirectional edge
            pairs = pairs[pairs['index_1'] < pairs['index_2']] # directional edge
            
            # Collect pairs for edges
            input_edges.append(pairs[['index_1', 'index_2']])

    return input_edges


def get_layerwise_graph (hits, filtering=True, inneredges=True):
    """
    Get input graph (edges) in adjacent layers and sectors with or without inner edges.

    Parameters
    ----------
    hits : pd.DataFrame
        hits information
    filtering : bool, optional
        if True, graph construction in adjacent sectors
        if False, do not require any sector constraint
    inneredges : bool, optional
        if True, include edges within a layer (samelayer edges)
        if False, do not include samelayer edges
    """
    # construct layerwise graph
    if inneredges:
        input_graph = extend_layerwise_graph_with_inner_edges(hits, filtering)
    else:
        input_graph = construct_layerwise_graph(hits, filtering)

    # concatenate and transform
    input_graph = pd.concat(input_graph).to_numpy().T

    return input_graph









