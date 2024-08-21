#!/bin/bash

# evaluate an event for particular epsilon.

if [ $# -lt 1 ]; then
  echo -e "First Argument is Needed.\n"
  echo -e "USAGE: ./eval.sh <event_id> <epsilon_cut> <max_events>"
  exit 1
fi

# params
evtid=1
maxevts=100
epsilon=0.25
edge_score_cut=0.0

# input
if test "$1" != ""; then
  evtid=$1
fi

if test "$2" != ""; then
  epsilon=$2
fi

if test "$3" != ""; then
  maxevts=$3
fi


# fractions
fraction=0.5

if (( $(echo "$fraction == 0.5" | bc -l) )); then
  fraction=$(echo "$fraction + 0.00001" | bc -l)
fi

echo "Running Track Building with Max Events: {$maxevts} and Epsilon: {$epsilon}"

# Data Directories
inputdir="run/gnn_evaluation/test"
reco_trkx_dir="run/trkx_from_gnn"
mkdir -p $reco_trkx_dir

# Tracks from GNN
python trkx_from_gnn.py \
    --input-dir $inputdir \
    --output-dir $reco_trkx_dir \
    --max-evts $maxevts \
    --num-workers 8 \
    --score-name "score" \
    --edge-score-cut $edge_score_cut \
    --epsilon $epsilon \
    --min-samples 2


echo "Running Evaluation for Event $evtid\n"



# Data Directories
eval_trkx_dir="run/trkx_reco_eval/$1"

# Evaluate Reco. Tracks
python eval_reco_trkx.py \
    --csv-path $inputdir \
    --reco-track-path $reco_trkx_dir \
    --outname $eval_trkx_dir \
    --event-id $evtid \
    --num-workers 8 \
    --force \
    --min-pt 0.0 \
    --min-hits-truth 7 \
    --min-hits-reco 6 \
    --frac-reco-matched $fraction \
    --frac-truth-matched $fraction

