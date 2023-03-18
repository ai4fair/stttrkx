#!/bin/bash

# This script combines scripts 'trkx_from_gnn.py' and 'trkx_reco_eval.py' together.
# To run these scripts individually, see 'trkx_from_gnn.sh' and 'trkx_reco_eval.sh'.

# max events
maxevts=30000

# trkx_from_gnn
inputdir="../run_all/fwp_gnn_processed/pred"
outputdir="../run_all/fwp_gnn_segmenting/seg"

# trkx_reco_eval
raw_tracks_path="../run_all/fwp_gnn_processed/pred"
reco_tracks_path="../run_all/fwp_gnn_segmenting/seg"
outputdir_eval="../run_all/fwp_gnn_segmenting/epsilon"
outfile=$outputdir_eval"/eps"
mkdir -p $outputdir_eval

# fractions
fraction=0.5

if (( $(echo "$fraction == 0.5" | bc -l) )); then
  fraction=$(echo "$fraction + 0.00001" | bc -l)
fi

# Search DBSCAN Epsilon (0.25 found best)
epsilons=(0.015 0.025 0.050 0.075 0.1 0.15 0.2 0.25 0.35 0.45 0.55 0.75 0.85 0.95 1.0)


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
        --csv-path $raw_tracks_path \
        --reco-track-path $reco_tracks_path \
        --outname $outfile \
        --max-evts $maxevts \
        --num-workers 8 \
        --force \
        --min-pt 0.0 \
        --min-hits-truth 7 \
        --min-hits-reco 6 \
        --frac-reco-matched $fraction \
        --frac-truth-matched $fraction
done




