#!/bin/bash

# This script runs 'trkx_from_gnn.py'

# Max Events
maxevts=5000

# Data Directories
inputdir="run/gnn_evaluation/test"
outputdir="run/trkx_from_gnn"
mkdir -p $outputdir

# Tracks from GNN
python trkx_from_gnn.py \
    --input-dir $inputdir \
    --output-dir $outputdir \
    --max-evts $maxevts \
    --num-workers 8 \
    --score-name "score" \
    --edge-score-cut 0.0 \
    --epsilon 0.25 \
    --min-samples 2
