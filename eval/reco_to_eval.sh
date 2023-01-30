#!/bin/bash

# Combined run of trkx_from_gnn.py (trkx_from_gnn.sh)
# and eval_reco_trkx.py (eval_reco_trkx.sh) scripts.


# params
maxevts=5000
epsilon=0.25
edge_score_cut=0.0

# input
if test "$1" != ""; then
  maxevts=$1
fi

if test "$2" != ""; then
  epsilon=$2
fi

if test "$3" != ""; then
  edge_score_cut=$3
fi


# Data Directories
inputdir="../run_all/dnn_processed/test"     # input from gnn_processed/test or pred/
outputdir="../run_all/dnn_segmenting/seg"    # output of trkx_from_gnn.sh to gnn_segmenting/seg
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


# Data Directories
raw_inputdir=$inputdir
rec_inputdir=$outputdir
outputdir="../run_all/dnn_segmenting/eval/"  # output of eval_reco_trkx.sh
mkdir -p $outputdir

# Evaluate Reco. Tracks
python eval_reco_trkx.py \
    --raw-tracks-path $raw_inputdir \
    --reco-tracks-path $rec_inputdir \
    --outname $outputdir \
    --max-evts $maxevts \
    --force \
    --min-hits-truth 7 \
    --min-hits-reco 4 \
    --min-pt 0. \
    --frac-reco-matched 0.5 \
    --frac-truth-matched 0.5
