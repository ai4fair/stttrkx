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


# fractions
fraction=0.5

if (( $(echo "$fraction == 0.5" | bc -l) )); then
  fraction=$(echo "$fraction + 0.00001" | bc -l)
fi

python eval_reco_trkx.py \
    --csv-path $raw_inputdir \
    --reco-track-path $rec_inputdir \
    --outname $outfile \
    --event-id $evt \
    --num-workers 8 \
    --force \
    --min-pt 0.0 \
    --min-hits-truth 7 \
    --min-hits-reco 6 \
    --frac-reco-matched $fraction \
    --frac-truth-matched $fraction
