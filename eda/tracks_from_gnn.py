#!/usr/bin/env python
"""Get track candidates using GNN score. The script runs after 'eval_gnn_tf.py' script."""

import time
import os
import glob
from multiprocessing import Pool
from functools import partial

import numpy as np
import scipy as sp
from sklearn.cluster import DBSCAN
import pandas as pd

import torch


def tracks_from_gnn(hit_id, score, senders, receivers,
    edge_score_cut=0., epsilon=0.25, min_samples=2, **kwargs):

    n_nodes = hit_id.shape[0]
    if edge_score_cut > 0:
        cuts = score > edge_score_cut
        score, senders, receivers = score[cuts], senders[cuts], receivers[cuts]
        
    # prepare the DBSCAN input, which the adjancy matrix with its value being the edge socre.
    e_csr = sp.sparse.csr_matrix((score, (senders, receivers)),
        shape=(n_nodes, n_nodes), dtype=np.float32)
    # rescale the duplicated edges
    e_csr.data[e_csr.data > 1] = e_csr.data[e_csr.data > 1]/2.
    # invert to treat score as an inverse distance
    e_csr.data = 1 - e_csr.data
    # make it symmetric
    e_csr_bi = sp.sparse.coo_matrix(
        (np.hstack([e_csr.tocoo().data, e_csr.tocoo().data]), 
        np.hstack([np.vstack([e_csr.tocoo().row, e_csr.tocoo().col]),                                                                   
        np.vstack([e_csr.tocoo().col, e_csr.tocoo().row])])))

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
    
def process(filename, torch_data_dir, outdir, score_name, **kwargs):
    evtid = int(os.path.basename(filename)[:-4])
    array = np.load(filename)
    score = array[score_name]
    senders = array['senders']
    receivers = array['receivers']

    torch_fname = os.path.join(torch_data_dir, "{:04}".format(evtid))
    data = torch.load(torch_fname, map_location='cpu')
    hit_id = data['hid'].numpy()

    predicted_tracks = tracks_from_gnn(hit_id, score, senders, receivers, **kwargs)

    # save reconstructed tracks into a file
    np.savez(
        os.path.join(outdir, "{}.npz".format(evtid)),
        predicts=predicted_tracks,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="construct tracks from the input created by eval_gnn_tf.py")
    add_arg = parser.add_argument
    # bookkeeping
    add_arg("--gnn-output-dir", help='directory saving results from evaluating GNNs', required=True)
    add_arg("--torch-data-dir", help='torch data directory', required=True)
    add_arg("--outdir", help='output file directory for track candidates', required=True)
    add_arg("--max-evts", help='maximum number of events for testing', type=int, default=1)
    add_arg("--num-workers", help='number of threads', default=1, type=int)
    add_arg("--score-name", help="score of edges, either GNN score or truth",
            default='score', choices=['score', 'truth'])

    # hyperparameters for DB scan
    add_arg("--edge-score-cut", help='edge score cuts', default=0, type=float)
    add_arg("--epsilon", help="epsilon in DBScan", default=0.25, type=float)
    add_arg("--min-samples", help='minimum number of samples in DBScan', default=2, type=int)

    args = parser.parse_args()

    inputdir = args.gnn_output_dir
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    all_files = glob.glob(os.path.join(inputdir, "*.npz"))
    n_tot_files = len(all_files)
    max_evts = args.max_evts if args.max_evts > 0 and args.max_evts <= n_tot_files else n_tot_files
    print("Out of {} events processing {} events with {} workers".format(n_tot_files, max_evts, args.num_workers))

    with Pool(args.num_workers) as p:
        process_fnc = partial(process, **vars(args))
        p.map(process_fnc, all_files[:max_evts])
