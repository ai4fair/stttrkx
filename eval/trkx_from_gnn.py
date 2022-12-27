#!/usr/bin/env python
# coding: utf-8

"""Modified version of 'tracks_from_gnn.py' (runs after 'eval_gnn_tf.py') script from the
exatrkx-iml2020. The code breakdown of the script is given in 'stt5_trkx.ipynb' notebook."""

import os
import glob
import torch

import scipy as sp
import numpy as np
import pandas as pd
import uproot

from multiprocessing import Pool
from functools import partial
from sklearn.cluster import DBSCAN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def tracks_from_gnn(hit_id, score, senders, receivers,
                    edge_score_cut=0., epsilon=0.25, min_samples=2,
                    **kwargs):
    
    """Use DBSCAN to build tracks after GNN stage. Important variables are
    edge pair (sender, receiver) and edge score plus other variables."""

    n_nodes = hit_id.shape[0]
    if edge_score_cut > 0:
        cuts = score > edge_score_cut
        score, senders, receivers = score[cuts], senders[cuts], receivers[cuts]
        
    # prepare the DBSCAN input, which the adjancy matrix with its value being the edge socre.
    e_csr = sp.sparse.csr_matrix((score, (senders, receivers)), shape=(n_nodes, n_nodes), dtype=np.float32)
    
    # rescale the duplicated edges
    e_csr.data[e_csr.data > 1] = e_csr.data[e_csr.data > 1]/2.
    
    # invert to treat score as an inverse distance
    e_csr.data = 1 - e_csr.data
    
    # make it symmetric
    e_csr_bi = sp.sparse.coo_matrix(
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


def process(filename, output_dir, score_name, **kwargs):
    """prepare a multiprocessing function for track building"""
    
    # get the event_id from the filename
    # evtid = int(os.path.basename(filename))  # [:-4] was to skip .npz extension, skipped in my case.
    evtid = int(os.path.basename(filename))
    
    # gnn_prcessed data by GNNBuilder Callback
    gnn_data = torch.load(filename, map_location=device)
    score = gnn_data.scores[:gnn_data.edge_index.shape[1]]  # score has twice the size of edge_index (flip(0) was used)
    senders = gnn_data.edge_index[0]
    receivers = gnn_data.edge_index[1]
    
    # TODO: Its better to have TClonesArray Index for STTHit as "gnn_data.cid". To make 
    # the code work without complicated modifications, just replace hit_id = gnn_data.cid
    hit_id = gnn_data.hid
    
    # predicted tracks from the GNN stage
    predicted_tracks = tracks_from_gnn(hit_id, score, senders, receivers, **kwargs)
 
    # save reconstructed tracks into a file
    
    # (i) DataFrame to Torch
    # torch.save(predicted_tracks, os.path.join(output_dir, "{}.pt".format(evtid)))
    output_file = os.path.join(output_dir, "{}.pt".format(evtid))
    with open(output_file, "wb") as pickle_file:
        torch.save(predicted_tracks, pickle_file)
    
    # (ii) DataFrame to CSV
    output_file = os.path.join(output_dir, "{}.csv".format(evtid))
    predicted_tracks.to_csv(output_file, index=False)
    
    # (iii) DataFrame to ROOT
    output_file = os.path.join(output_dir, "{}.root".format(evtid))
    with uproot.recreate(output_file) as root_file:
        root_file["TrackML"] = predicted_tracks
    
    
if __name__ == "__main__":
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
    inputdir = args.input_dir        # GNN evaluation (Scores produced after test_step of GNN Stage.)
    outputdir = args.output_dir      # Track building (cols=['hit_id', 'track_id'] 
    print("inputdir: ", inputdir)
    print("outputdir: ", outputdir)
    os.makedirs(outputdir, exist_ok=True)
        
    all_files = glob.glob(os.path.join(inputdir, "*"))
    all_files = sorted(all_files)
    
    n_tot_files = len(all_files)
    max_evts = args.max_evts if 0 < args.max_evts <= n_tot_files else n_tot_files
    print("Out of {} events processing {} events with {} workers.\n".format(n_tot_files, max_evts, args.num_workers))
    
    
    # TODO: Use UpROOT to Create File to Save Tracks as TTrees. Due to Multiprocessing
    # I can only creat ROOT files event by event. So I moved this code to "process()".
    # If we have a single trackml.root, then differnt worker tries to write it at the 
    # same time. So only solution is creat files event by event.
        
    # with uproot.open(os.path.join(outputdir, "/trackml.root")) as root_file:
    #    print("ROOT created at ", root_file)
    # root_file = uproot.open(os.path.join(outputdir, "trackml.root"))
    # root_file.close()
        
    
    # Multiprocess to Build Tracks
    with Pool(args.num_workers) as p:
        process_fnc = partial(process, **vars(args))
        p.map(process_fnc, all_files[:max_evts])
