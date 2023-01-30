#!/bin/bash

# This script runs 'trkx_reco_eval.py', the trkx_from_gnn.py must
# run before this script. Better way is to use 'reco_to_eval.sh'.

# Max Events
maxevts=5000

if test "$1" != ""; then
  maxevts=$1
fi


# Data Directories
raw_inputdir="../run_all/dnn_processed/test"     # output of DNN stage in test/pred
rec_inputdir="../run_all/dnn_segmenting/seg"     # output of trkx_from_gnn.sh 
   outputdir="../run_all/dnn_segmenting/eval/"   # output of eval_reco_trkx.sh
mkdir -p $outputdir

# Evaluate Reco. Tracks
python eval_reco_trkx.py \
    --reco-tracks-path $rec_inputdir \
    --raw-tracks-path $raw_inputdir \
    --outname $outputdir \
    --max-evts $maxevts \
    --force \
    --min-hits-truth 7 \
    --min-hits-reco 5 \
    --min-pt 0. \
    --frac-reco-matched 0.7 \
    --frac-truth-matched 0.7
