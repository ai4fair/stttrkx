#!/bin/bash

# params
max_events=20000
edge_score_cut_min=0.8
edge_score_cut_max=0.8
edge_score_cut_step=0.1
workers=4
fraction=0.5
original_fraction=$fraction

# Don't move above outfile, name will be messed up.
if (($(echo "$fraction == 0.5" | bc -l))); then
  fraction=$(echo "$fraction + 0.00001" | bc -l)
fi

# Data Directories
input_dir="/mnt/data1/user/n_inde01/machineLearning/XiAntiXi/classification/trained_from_scratch/test"
output_dir="/mnt/data1/user/n_inde01/machineLearning/XiAntiXi/evaluation/trained_from_scratch"

edge_score_cut=$edge_score_cut_min
while (($(echo "$edge_score_cut <= $edge_score_cut_max" | bc -l))); do
  epsilon=$(echo "1.0 - $edge_score_cut" | bc -l)

  echo "Running with edge_score_cut: $edge_score_cut and epsilon: $epsilon"

  # Tracks from GNN
  python trkx_from_gnn.py \
    --input-dir "$input_dir" \
    --output-dir "$output_dir"/events \
    --max-evts $max_events \
    --num-workers $workers \
    --score-name "scores" \
    --edge-score-cut "$edge_score_cut" \
    --epsilon "$epsilon" \
    --min-samples 2

  # Evaluate Reco. Tracks
  python eval_reco_trkx.py \
    --csv-path "$input_dir" \
    --reco-track-path "$output_dir"/events \
    --outname "$output_dir"/summaries/"$original_fraction"_"$edge_score_cut" \
    --max-evts $max_events \
    --num-workers $workers \
    --force \
    --min-pt 0.0 \
    --min-hits-truth 7 \
    --min-hits-reco 6 \
    --frac-reco-matched "$fraction" \
    --frac-truth-matched "$fraction"

  edge_score_cut=$(echo "$edge_score_cut + $edge_score_cut_step" | bc -l)

done
