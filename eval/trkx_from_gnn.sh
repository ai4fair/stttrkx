#!/bin/bash

# This script runs 'trkx_from_gnn.py'

maxevts=5000
inputdir="run/gnn_evaluation/test"
outputdir="run/trkx_from_gnn"

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
