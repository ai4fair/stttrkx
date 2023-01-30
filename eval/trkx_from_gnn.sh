#!/bin/bash

# This script runs 'trkx_from_gnn.py'

# params
epsilon=0.25
maxevts=5000
edge_score_cut=0.5


# input
if test "$1" != ""; then
  epsilon=$1
fi

if test "$2" != ""; then
  maxevts=$2
fi

if test "$3" != ""; then
  edge_score_cut=$3
fi


# Data Directories
inputdir="../run_all/dnn_processed/test"  # input from gnn_processed/test or pred/
outputdir="../run_all/dnn_segmenting/seg" # output of trkx_from_gnn.sh, track candidates
mkdir -p $outputdir

# Tracks from GNN
python trkx_from_gnn.py \
    --input-dir $inputdir \
    --output-dir $outputdir \
    --max-evts $maxevts \
    --num-workers 8 \
    --score-name "score" \
    --edge-score-cut $edge_score_cut \
    --epsilon $epsilon \
    --min-samples 2
