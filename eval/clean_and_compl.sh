#!/bin/bash

# Combined run of trkx_from_gnn.py (trkx_from_gnn.sh)
# and eval_reco_trkx.py (eval_reco_trkx.sh) scripts.


# Max Events
maxevts=5000

# Reco. Event Types
evt_type="clean"


# Data Directories
inputdir="run/gnn_evaluation/test"
reco_trkx_dir="run/trkx_from_gnn/"$evt_type
eval_trkx_dir="run/trkx_reco_eval/"$evt_type

# Evaluate Reco. Tracks
python eval_reco_trkx.py \
    --reco-tracks-path $reco_trkx_dir \
    --raw-tracks-path $inputdir \
    --outname $eval_trkx_dir \
    --max-evts $maxevts \
    --force \
    --min-hits-truth 7 \
    --min-hits-reco 4 \
    --min-pt 0. \
    --frac-reco-matched 0.5 \
    --frac-truth-matched 0.5
