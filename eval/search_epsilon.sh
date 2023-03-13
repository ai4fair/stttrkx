#!/bin/bash

# This script combines scripts 'trkx_from_gnn.py' and 'trkx_reco_eval.py' together.
# To run these scripts individually, see 'trkx_from_gnn.sh' and 'trkx_reco_eval.sh'.

# max events
maxevts=10000

# trkx_from_gnn
inputdir="../run_all/fwp_gnn_processed/pred"
outputdir="../run_all/fwp_gnn_segmenting/seg"

# trkx_reco_eval
raw_tracks_path="../run_all/fwp_gnn_processed/pred"
reco_tracks_path="../run_all/fwp_gnn_segmenting/seg"
outputdir_eval="../run_all/fwp_gnn_segmenting/epsilon"
outfile=$outputdir_eval"/all"
mkdir -p $outputdir_eval

# Search DBSCAN Epsilon (0.25 found best)
epsilons=(0.1 0.15 0.2 0.25 0.35 0.45 0.55 0.75 0.85 0.95 1.0)
 
for t in ${epsilons[@]}; do
    echo "epsilon: $t"
    
    # reco tracks from GNN
    python trkx_from_gnn.py \
        --input-dir $inputdir \
        --output-dir $outputdir \
        --max-evts $maxevts \
        --num-workers 8 \
        --score-name "scores" \
        --edge-score-cut 0.0 \
        --epsilon $t \
        --min-samples 2
    
    # evaluate reco tracks from GNN
    python eval_reco_trkx.py \
        --reco-tracks-path $reco_tracks_path \
        --raw-tracks-path $raw_tracks_path \
        --outname $outfile \
        --max-evts $maxevts \
        --force \
        --min-hits-truth 7 \
        --min-hits-reco 4 \
        --min-pt 0. \
        --frac-reco-matched 0.5 \
        --frac-truth-matched 0.5
done




