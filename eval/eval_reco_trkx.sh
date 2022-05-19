#!/bin/bash

# This script combines scripts 'trkx_from_gnn.py' and 'trkx_reco_eval.py' together.
# To run these scripts individually, see 'trkx_from_gnn.sh' and 'trkx_reco_eval.sh'.

# max events
maxevts=100

# trkx_from_gnn.py
inputdir="run/gnn_evaluation/test"
outputdir="run/trkx_from_gnn"

# trkx_reco_eval.py
reco_tracks_path="run/trkx_from_gnn"
raw_tracks_path="run/gnn_evaluation/test"
outputdir_eval="run/trkx_reco_eval/eval"

# reco tracks from GNN
python trkx_from_gnn.py \
    --input-dir $inputdir \
    --output-dir $outputdir \
    --max-evts $maxevts \
    --num-workers 8 \
    --score-name "score" \
    --edge-score-cut 0.0 \
    --epsilon 0.25 \
    --min-samples 2

# evaluate reco tracks from GNN
python trkx_reco_eval.py \
    --reco-tracks-path $reco_tracks_path \
    --raw-tracks-path $raw_tracks_path \
    --outname $outputdir_eval \
    --max-evts $maxevts \
    --force \
    --min-hits-truth 7 \
    --min-hits-reco 4 \
    --min-pt 0. \
    --frac-reco-matched 0.5 \
    --frac-truth-matched 0.5
