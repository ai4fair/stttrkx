#!/bin/bash

# This script combines scripts 'trkx_from_gnn.py' and 'trkx_reco_eval.py' together.
# To run these scripts individually, see 'trkx_from_gnn.sh' and 'trkx_reco_eval.sh'.

# max events
max_events=100000

# matching fraction for the evaluation
fractions=(0.5 0.75 0.95)

# minimum transversal momentum
min_pt=0.0

# minimum number of true hits for a reconstructible track
min_true_hits=6

# minimum number of reconstructed hits for a reconstructed track
# currently all true hits should be reconstructed
min_reco_hits=6

# workers for multiprocessing
num_workers=8

# trkx_from_gnn
input_dir="/mnt/data1/user/n_inde01/machineLearning/XiAntiXi/classification/trained_from_scratch/test/"
output_dir="/mnt/data1/user/n_inde01/machineLearning/XiAntiXi/evaluation/trained_from_scratch"

# Search DBSCAN epsilon which is effectively a cut on the edge score (1-epsilon) to define node neighborhoods.
epsilons=(0.015 0.025 0.050 0.075 0.1 0.15 0.2 0.25 0.35 0.45 0.55 0.75 0.85 0.95 1.0)

for epsilon in "${epsilons[@]}"; do
    echo "epsilon: $epsilon"

    rec_track_dir="$output_dir/eps_${epsilon}/events/"

    mkdir -p "$rec_track_dir"
    
    # reco tracks from GNN
    python trkx_from_gnn.py \
        --input-dir $input_dir \
        --output-dir "$rec_track_dir" \
        --max-evts $max_events \
        --num-workers $num_workers \
        --score-name "scores" \
        --edge-score-cut 0.0 \
        --epsilon "$epsilon" \
        --min-samples 2
    
    for fraction in "${fractions[@]}"; do
        
        echo "  fraction: $fraction"

        summary_dir="$output_dir/eps_${epsilon}/summaries/"
        mkdir -p "$summary_dir"

        if (( $(echo "$fraction == 0.5" | bc -l) )); then
          fraction=$(echo "$fraction + 0.00001" | bc -l)
        fi

        # evaluate reco tracks from GNN
        python eval_reco_trkx.py \
            --csv-path $input_dir \
            --reco-track-path "$rec_track_dir" \
            --outname "$summary_dir"/mf_"${fraction}" \
            --max-evts $max_events \
            --num-workers $num_workers \
            --force \
            --min-pt $min_pt \
            --min-hits-truth $min_true_hits \
            --min-hits-reco $min_reco_hits \
            --frac-reco-matched "$fraction" \
            --frac-truth-matched "$fraction"
    done
done




