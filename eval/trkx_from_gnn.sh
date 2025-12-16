#!/bin/bash

# This script runs 'trkx_from_gnn.py'

# params
epsilon=0.2
max_events=100_000

# input
if test "$1" != ""; then
  epsilon=$1
fi

if test "$2" != ""; then
  max_events=$2
fi

# Data Directories
inputdir="/mnt/data1/user/n_inde01/machineLearning/XiAntiXi/classification/trained_from_muon/test" # input from GNN stage as in test/pred
outputdir="/mnt/data1/user/n_inde01/machineLearning/XiAntiXi/evaluation/trained_from_muon_0.75mf"   # output of trkx_from_gnn.sh i.e. TrackCands
mkdir -p $outputdir

# original: trkx_from_gnn_v1
# uproot  : trkx_from_gnn_uproot
# cleaned : trkx_from_gnn

# Tracks from GNN
python trkx_from_gnn.py \
  --input-dir $inputdir \
  --output-dir $outputdir \
  --max-evts $max_events \
  --num-workers 10 \
  --score-name "scores" \
  --edge-score-cut 0 \
  --epsilon $epsilon \
  --min-samples 2
