#!/bin/bash

# Combined run of trkx_from_gnn.py (trkx_from_gnn.sh)
# and eval_reco_trkx.py (eval_reco_trkx.sh) scripts.


# Max Events
maxevts=5000

# Data Directories
inputdir="run/gnn_evaluation/test"
reco_trkx_dir="run/trkx_from_gnn"
mkdir -p $reco_trkx_dir

# Tracks from GNN
python trkx_from_gnn.py \
    --input-dir $inputdir \
    --output-dir $reco_trkx_dir \
    --max-evts $maxevts \
    --num-workers 8 \
    --score-name "score" \
    --edge-score-cut 0.0 \
    --epsilon 0.25 \
    --min-samples 2


# Data Directories
eval_trkx_dir="run/trkx_reco_eval/eval"
mkdir -p $eval_trkx_dir

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
