#!/bin/sh

maxevts=100
gnn_pred="run/gnn_evaluation/test"
reco_tracks="run/trkx_from_gnn"
outputdir="run/trkx_reco_eval/eval"

# good and bad events
#gnn_pred="run/gnn_evaluation/test_bad"
#reco_tracks="run/trkx_from_gnn_bad"
#outputdir="run/trkx_reco_eval/bad"

python trkx_reco_eval.py \
    --reco-tracks-path $reco_tracks \
    --raw-tracks-path $gnn_pred \
    --outname $outputdir \
    --max-evts $maxevts \
    --force \
    --min-hits-truth 7 \
    --min-hits-reco 4 \
    --min-pt 0. \
    --frac-reco-matched 0.5 \
    --frac-truth-matched 0.5
