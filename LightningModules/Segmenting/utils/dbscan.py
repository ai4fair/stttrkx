#!/usr/bin/env python
# coding: utf-8

# TODO: Refined version of `eval/trkx_from_gnn.py` script meant to be called from `src/ccl`.

"""Modified version of 'tracks_from_gnn.py' (runs after 'eval_gnn_tf.py') script from the
exatrkx-iml2020. The code breakdown of the script is given in 'stt5_trkx.ipynb' notebook."""

import os
import glob
import torch
import logging
import numpy as np
import pandas as pd
import scipy.sparse as sp

from multiprocessing import Pool
from functools import partial
from sklearn.cluster import DBSCAN

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def label_graph_dbscan(
        input_file: str, output_dir: str, edge_cut: float = 0., overwrite: bool = True, **kwargs
) -> None:

    """prepare a multiprocessing function for track building"""
    
    try:

        output_file = os.path.join(output_dir, os.path.split(input_file)[-1])
        print(output_file)

        if not os.path.exists(output_file) or overwrite:

            logging.info("Preparing event {}".format(output_file))
            graph = torch.load(input_file, map_location="cpu")
            

            # evtid = int(os.path.basename(input_file))
            score = graph.scores[:graph.edge_index.shape[1]]  # score has twice the size of edge_index (due to flip(0))
            senders = graph.edge_index[0]
            receivers = graph.edge_index[1]
            hit_id = graph.hid
            
            n_nodes = hit_id.shape[0]
            if edge_cut > 0:
                cuts = score > edge_cut
                score, senders, receivers = score[cuts], senders[cuts], receivers[cuts]

            # prepare the DBSCAN input, adjancy matrix with its data being the edge socre.
            e_csr = sp.csr_matrix(
                (score, (senders, receivers)),
                shape=(n_nodes, n_nodes),
                dtype=np.float32)

            # rescale the duplicated edges
            e_csr.data[e_csr.data > 1] = e_csr.data[e_csr.data > 1] / 2.

            # invert to treat score as an inverse distance
            e_csr.data = 1 - e_csr.data

            # make it symmetric
            e_csr_bi = sp.coo_matrix(
                (np.hstack([e_csr.tocoo().data, e_csr.tocoo().data]),
                 np.hstack([np.vstack([e_csr.tocoo().row, e_csr.tocoo().col]),
                            np.vstack([e_csr.tocoo().col, e_csr.tocoo().row])]
                           )
                 )
            )

            # DBSCAN get track candidates
            clustering = DBSCAN(
                eps=epsilon, metric='precomputed',
                min_samples=min_samples).fit_predict(e_csr_bi)
            
            # FIXME (Done): add labels to graph and save it, same as label_graph_ccl
            # attach labels to data
            graph.labels = clustering[np.unique(e_csr_bi.tocoo().row)]
            
            # Convert data as hit_id, track_id dataframe
            track_labels = np.vstack(
                [np.unique(e_csr_bi.tocoo().row),
                 clustering[np.unique(e_csr_bi.tocoo().row)]])

            track_labels = pd.DataFrame(track_labels.T)
            track_labels.columns = ["hit_id", "track_id"]

            new_hit_id = np.apply_along_axis(
                lambda x: hit_id[x], 0, track_labels.hit_id.values)

            predicted_tracks = pd.DataFrame.from_dict(
                {"hit_id": new_hit_id, "track_id": track_labels.track_id})
            
            # FIXME: How about adding predicted_tracks as torch tensor in graph?
            graph.reco_tracks = torch.Tensor(predicted_tracks.values)
            
            with open(output_file, "wb") as pickle_file:
                torch.save(graph, pickle_file)

        else:
            logging.info("{} already exists".format(output_file))

    except Exception as inst:
        print("File:", input_file, "had exception", inst)
        


def tracks_from_cc(
    hit_id, senders, receivers, scores, edge_cut=0., epsilon=0.25, min_samples=2, **kwargs
):
    """Use DBSCAN to build tracks after GNN stage. Important variables are
    edge pair 'sender', 'receiver' and 'edge_score', plus other variables."""

    n_nodes = hit_id.shape[0]
    if edge_cut > 0:
        cuts = scores > edge_cut
        scores, senders, receivers = scores[cuts], senders[cuts], receivers[cuts]

    # prepare the DBSCAN input, which the adjancy matrix with its value being the edge socre.
    e_csr = sp.csr_matrix(
        (scores, (senders, receivers)),
        shape=(n_nodes, n_nodes),
        dtype=np.float32)

    # rescale the duplicated edges
    e_csr.data[e_csr.data > 1] = e_csr.data[e_csr.data > 1] / 2.

    # invert to treat edge score as an inverse distance
    e_csr.data = 1 - e_csr.data

    # make it symmetric
    e_csr_bi = sp.coo_matrix(
        (np.hstack([e_csr.tocoo().data, e_csr.tocoo().data]),
         np.hstack([np.vstack([e_csr.tocoo().row, e_csr.tocoo().col]),
                    np.vstack([e_csr.tocoo().col, e_csr.tocoo().row])]
                   )
         )
    )

    # DBSCAN get track candidates
    clustering = DBSCAN(
        eps=epsilon, metric='precomputed',
        min_samples=min_samples).fit_predict(e_csr_bi)

    track_labels = np.vstack(
        [np.unique(e_csr_bi.tocoo().row),
         clustering[np.unique(e_csr_bi.tocoo().row)]])

    track_labels = pd.DataFrame(track_labels.T)
    track_labels.columns = ["hit_id", "track_id"]

    new_hit_id = np.apply_along_axis(
        lambda x: hit_id[x], 0, track_labels.hit_id.values)

    tracks = pd.DataFrame.from_dict(
        {"hit_id": new_hit_id, "track_id": track_labels.track_id})

    return tracks


def process(filename, output_dir, **kwargs):
    """prepare a multiprocessing function for track building"""

    # get the event_id from the filename
    # evtid = int(os.path.basename(filename))  # [:-4] was to skip .npz extension, skipped in my case.
    evtid = int(os.path.basename(filename))

    # gnn_prcessed data by GNNBuilder Callback
    gnn_data = torch.load(filename, map_location=device)
    scores = gnn_data.scores[:gnn_data.edge_index.shape[1]]  # score has twice the size of edge_index (flip(0) was used)
    senders = gnn_data.edge_index[0]
    receivers = gnn_data.edge_index[1]
    hit_id = gnn_data.hid

    # predicted tracks from the GNN stage
    predicted_tracks = tracks_from_cc(hit_id, senders, receivers, scores, **kwargs)

    # save reconstructed tracks into a file
    # PyTorch convention is to save tensors using .pt file extension
    # See https://pytorch.org/docs/stable/notes/serialization.html#preserve-storage-sharing
    # torch.save(predicted_tracks, os.path.join(output_dir, "{}.pt".format(evtid)))
    torch.save(predicted_tracks, os.path.join(output_dir, "{}".format(evtid)))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build Tracks after GNN Evaluation (GNN Test Step).")
    add_arg = parser.add_argument

    # Bookkeeping
    add_arg("--input-dir", help='Directory saving results from evaluating GNNs', required=True)

    # add_arg("--torch-data-dir", help='torch data directory', required=True)
    add_arg("--output-dir", help='Output file directory for track candidates.', required=True)
    add_arg("--max-evts", help='Maximum number of events for track building.', type=int, default=1)
    add_arg("--num-workers", help='Number of workers/threads.', default=8, type=int)
    add_arg("--score-name", help="Edge score, either GNN score or truth",
            default='score', choices=['score', 'truth'])

    # Hyperparameters for DBScan
    add_arg("--edge-score-cut", help='edge score cuts', default=0, type=float)
    add_arg("--epsilon", help="epsilon in DBScan", default=0.25, type=float)
    add_arg("--min-samples", help='minimum number of samples in DBScan', default=2, type=int)

    # Collect Arguments
    args = parser.parse_args()

    # Input/Output Data
    inputdir = args.input_dir  # GNN evaluation (Scores produced after test_step of GNN Stage.)
    outputdir = args.output_dir  # Track building (cols=['hit_id', 'track_id']
    print("inputdir: ", inputdir)
    print("outputdir: ", outputdir)
    os.makedirs(outputdir, exist_ok=True)

    all_files = glob.glob(os.path.join(inputdir, "*"))
    all_files = sorted(all_files)

    n_tot_files = len(all_files)
    max_evts = args.max_evts if 0 < args.max_evts <= n_tot_files else n_tot_files
    print("Out of {} events processing {} events with {} workers.\n".format(n_tot_files, max_evts, args.num_workers))

    # Multiprocess to Build Tracks
    with Pool(args.num_workers) as p:
        process_fnc = partial(process, **vars(args))
        p.map(process_fnc, all_files[:max_evts])


if __name__ == "__main__":
    main()
