#!/bin/bash

# This script runs 'trkx_reco_eval.py' for a single event.

# Max Events
evt=1

if test "$1" != ""; then
  evt=$1
fi


# Stage [dnn, gnn, agnn,...]
ann=gnn


# Data Directories
raw_inputdir="../run_all/fwp_"$ann"_processed/pred"  # output of GNN stage as in test/pred
rec_inputdir="../run_all/fwp_"$ann"_segmenting/seg"  # output of trkx_from_gnn.sh
outputdir="../run_all/fwp_"$ann"_segmenting/eval"    # output of eval_reco_trkx.sh
outfile=$outputdir"/$1"                              # name prefix of output files
mkdir -p $outputdir



evtid=$1
gnn_pred="run/gnn_evaluation/test"
reco_tracks="run/trkx_from_gnn"
outputdir="run/trkx_reco_eval/$1"

# good and bad events
#gnn_pred="run/gnn_evaluation/test_bad"
#reco_tracks="run/trkx_from_gnn_bad"
#outputdir="run/trkx_reco_eval/bad"

python eval_reco_trkx.py \
    --raw-tracks-path $raw_inputdir \
    --reco-tracks-path $rec_inputdir \
    --outname $outfile \
    --event-id $evt \
    --force \
    --min-pt 0.0 \
    --min-hits-truth 7 \
    --min-hits-reco 6 \
    --frac-reco-matched 0.5 \
    --frac-truth-matched 0.5
