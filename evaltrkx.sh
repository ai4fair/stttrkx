#!/bin/bash

maxevts=100

# trkx_from_gnn
inputdir="run/gnn_evaluation/test"
outputdir="run/trkx_from_gnn"

# trkx_reco_eval
reco_tracks_path="run/trkx_from_gnn"
raw_tracks_path="run/gnn_evaluation/test"
outputdir_eval="run/trkx_reco_eval/eval"

# DBSCAN Epsilon (0.25 found best)
epsilons=(0.1 0.15 0.2 0.25 0.35 0.45 0.55 0.75 0.85 0.95)
 
for t in ${epsilons[@]}; do
    echo "epsilon: $t"
    python trkx_from_gnn.py \
        --input-dir $inputdir \
        --output-dir $outputdir \
        --max-evts $maxevts \
        --num-workers 8 \
        --score-name "score" \
        --edge-score-cut 0.0 \
        --epsilon $t \
        --min-samples 2

    python trkx_reco_eval.py \
        --reco-tracks-path $reco_tracks_path \
        --raw-tracks-path $raw_tracks_path \
        --outname $outputdir_eval \
        --max-evts $maxevts \
        --force \
        --min-hits-truth 7 \
        --min-hits-reco 4 \
        --min-pt 0. \
        --frac-reco-matched 0.5 \
        --frac-truth-matched 0.5
done




