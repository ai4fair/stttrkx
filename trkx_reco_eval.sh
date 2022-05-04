#!/bin/sh

reco_tracks_path="run/trkx_from_gnn"
raw_tracks_path="run/gnn_evaluation/test"
outputdir="run/trkx_reco_eval/eval"
maxevts=100

python trkx_reco_eval.py \
    --reco-tracks-path $reco_tracks_path \
    --raw-tracks-path $raw_tracks_path \
    --outname $outputdir \
    --max-evts $maxevts \
    --force \
    --min-hits-truth 7 \
    --min-hits-reco 4 \
    --min-pt 0. \
    --frac-reco-matched 0.5 \
    --frac-truth-matched 0.5
