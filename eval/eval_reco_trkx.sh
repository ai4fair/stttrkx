#!/bin/bash

# This script runs 'trkx_reco_eval.py', the trkx_from_gnn.py must
# run before this script. Better way is to use 'reco_to_eval.sh'.

# Max Events
maxevts=5000

if test "$1" != ""; then
  maxevts=$1
fi


# Stage [dnn, gnn, agnn,...]
ann=dnn


# Data Directories
raw_inputdir="../run_all/"$ann"_processed/pred"  # output of GNN stage as in test/pred
rec_inputdir="../run_all/"$ann"_segmenting/seg"  # output of trkx_from_gnn.sh
outputdir="../run_all/"$ann"_segmenting/eval"    # output of eval_reco_trkx.sh
outfile=$outputdir"/all"                         # name prefix of output files
mkdir -p $outputdir


# Evaluate Reco. Tracks
python eval_reco_trkx.py \
    --reco-tracks-path $rec_inputdir \
    --raw-tracks-path $raw_inputdir \
    --outname $outfile \
    --max-evts $maxevts \
    --force \
    --min-pt 0. \
    --min-hits-truth 7 \
    --min-hits-reco 4 \
    --frac-reco-matched 0.5 \
    --frac-truth-matched 0.5

# Last 4 Params:

# ATLAS: 7    PANDA: ?
# ATLAS: 4    PANDA: 5
# ATLAS: 0.5  PANDA: 0.7-0.8
# ATLAS: 0.5  PANDA: 0.7-0.8

