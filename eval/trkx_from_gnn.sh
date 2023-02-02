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

# Stage [dnn, gnn, agnn,...]
ann=gnn

# Data Directories
inputdir="../run_all/"$ann"_processed/pred"  # input from GNN stage as in test/pred
outputdir="../run_all/"$ann"_segmenting/seg" # output of trkx_from_gnn.sh i.e. TrackCands
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

