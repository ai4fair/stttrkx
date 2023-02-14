#!/usr/bin/env python
# coding: utf-8

import os
import torch
import logging
import numpy as np
import scipy.sparse as sps
import scipy.sparse.csgraph as scigraph
from torch_geometric.utils import to_scipy_sparse_matrix

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Connected Component Labelling (CCL)
def ccl_labelling(input_file, output_dir, edge_cut=0.5, **kwargs):
    """Loads an input_file and outputs a segmented (i.e. labelled) graph. Function
    uses the to_scipy_sparse_matrix() from PyG library to build a sparse array."""

    try:

        output_file = os.path.join(output_dir, os.path.split(input_file)[-1])
        if not os.path.exists(output_file) or kwargs["overwrite"]:

            logging.info("Preparing event {}".format(output_file))
            graph = torch.load(input_file, map_location=device)
            
            # edge scores
            scores = graph.scores
            
            # half the length, gnn gives scores for bidirected graphs
            scores = scores[:graph.edge_index.shape[1]] 
            
            # apply edge score cut
            e_mask = scores > edge_cut
            
            # filter passing edges
            passing_edges = graph.edge_index[:, e_mask]

            # convert to sparse array
            sparse_edges = to_scipy_sparse_matrix(passing_edges, num_nodes=graph.x.size(0))
            
            # run connected components
            labels = sps.csgraph.connected_components(
                        sparse_edges, directed=False, return_labels=True
            )[1]

            # attach labels to data
            # graph.labels = torch.from_numpy(labels).long()  # .astype(np.int32)
            graph.labels = torch.from_numpy(labels).type_as(passing_edges)

            with open(output_file, "wb") as pickle_file:
                torch.save(graph, pickle_file)

        else:
            logging.info("{} already exists".format(output_file))

    except Exception as inst:
        print("File:", input_file, "had exception", inst)


def ccl_labelling_v2(input_file, output_dir, edge_cut=0.5, **kwargs):
    """Loads an input_file and outputs a segmented (i.e. labelled) graph. Function
    uses the sparse.coo_matrix() from the scipy library to build a sparse array."""

    try:

        output_file = os.path.join(output_dir, os.path.split(input_file)[-1])
        if not os.path.exists(output_file) or kwargs["overwrite"]:
            
            logging.info("Preparing event {}".format(output_file))
            graph = torch.load(input_file, map_location=device)
            
            # edge scores
            scores = graph.scores
            
            # half the length, gnn gives scores for bidirected graphs
            scores = scores[:graph.edge_index.shape[1]] 
            
            # apply edge score cut
            e_mask = scores > edge_cut
            
            # filter passing edges
            passing_edges = graph.edge_index[:, e_mask]
            
            # convert to sparse array
            num_nodes = graph.x.size(0)      # or graph.hid.shape[0]
            sparse_edges = sps.coo_matrix(
                (np.ones(passing_edges.shape[1]), passing_edges.cpu().numpy()),
                shape=(num_nodes, num_nodes),
            )
            
            # run connected components
            labels = sps.csgraph.connected_components(
                        sparse_edges, directed=False, return_labels=True
            )[1]
            
            # attach labels to data
            graph.labels = torch.from_numpy(labels).type_as(passing_edges)

            with open(output_file, "wb") as pickle_file:
                torch.save(graph, pickle_file)

        else:
            logging.info("{} already exists".format(output_file))

    except Exception as inst:
        print("File:", input_file, "had exception", inst)
