#!/usr/bin/env python
# coding: utf-8

import os
import torch
import logging
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as scigraph
from torch_geometric.utils import to_scipy_sparse_matrix

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def label_graph_ccl(
        input_file: str, output_dir: str, edge_cut: float = 0.5, overwrite: bool = True, **kwargs
) -> None:

    """Loads an input_file and outputs a segmented (i.e. labelled) graph.
    Args:
        input_file: Location of the input graph (a torch pickled file containing a PyG data object).
        output_dir: Location of labeled graphs as torch pickled file
        edge_cut: The minimum score for an edge to become part of a segment
        overwrite: Whether to overwrite existing files

    return:
        None
    """

    try:

        output_file = os.path.join(output_dir, os.path.split(input_file)[-1])

        if not os.path.exists(output_file) or overwrite:

            logging.info("Preparing event {}".format(output_file))
            graph = torch.load(input_file, map_location=device)
            
            # score has twice the size of edge_index (flip(0) was used)
            scores = graph.scores[:graph.edge_index.shape[1]] 
            
            # apply cut
            passing_edges = graph.edge_index[:, scores > edge_cut]

            # get connected components
            sparse_edges = to_scipy_sparse_matrix(passing_edges)
            labels = scigraph.connected_components(sparse_edges)[1]

            # attach labels to data
            graph.labels = torch.from_numpy(labels).type_as(passing_edges)

            with open(output_file, "wb") as pickle_file:
                torch.save(graph, pickle_file)

        else:
            logging.info("{} already exists".format(output_file))

    except Exception as inst:
        print("File:", input_file, "had exception", inst)


def label_graph(
        input_file: str, output_dir: str, edge_cut: float = 0.5, overwrite: bool = True, **kwargs
) -> None:
    """Loads an input_file and outputs a segmented (i.e. labelled) graph.
    Args:
        input_file: Location of the input graph (a torch pickled file containing a PyG data object).
        output_dir: Location of labeled graphs as torch pickled file
        edge_cut: The minimum score for an edge to become part of a segment
        overwrite: Whether to overwrite existing files

    return:
        None
    """

    try:

        output_file = os.path.join(output_dir, os.path.split(input_file)[-1])

        if not os.path.exists(output_file) or overwrite:

            logging.info("Preparing event {}".format(output_file))
            graph = torch.load(input_file, map_location=device)

            # score has twice the size of edge_index (flip(0) was used)
            scores = graph.scores[:graph.edge_index.shape[1]] 
            
            # apply cut
            passing_edges = graph.edge_index[:, scores > edge_cut]

            # attach labels to data
            graph.labels = label_segments(passing_edges, len(graph.x))

            with open(output_file, "wb") as pickle_file:
                torch.save(graph, pickle_file)

        else:
            logging.info("{} already exists".format(output_file))

    except Exception as inst:
        print("File:", input_file, "had exception", inst)


def label_segments(input_edges, num_nodes):

    # get connected components
    # sparse_edges = to_scipy_sparse_matrix(input_edges)
    
    sparse_edges = sp.coo_matrix(
        (np.ones(input_edges.shape[1]), input_edges.cpu().numpy()),
        shape=(num_nodes, num_nodes),
    )
    connected_components = scigraph.connected_components(sparse_edges)[1]

    return torch.from_numpy(connected_components).type_as(input_edges)
