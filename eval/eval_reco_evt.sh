#!/bin/bash

# This script runs 'trkx_reco_eval.py' for a single event.

if [ $# -lt 1 ]; then
  echo -e "Provide Event ID.\n"
  echo -e "USAGE: ./eval.sh <event_id>"
  exit 1
fi


evtid=$1
gnn_pred="run/gnn_evaluation/test"
reco_tracks="run/trkx_from_gnn"
outputdir="run/trkx_reco_eval/$1"

# good and bad events
#gnn_pred="run/gnn_evaluation/test_bad"
#reco_tracks="run/trkx_from_gnn_bad"
#outputdir="run/trkx_reco_eval/bad"

python eval_reco_trkx.py \
    --reco-tracks-path $reco_tracks \
    --raw-tracks-path $gnn_pred \
    --outname $outputdir \
    --event-id $evtid \
    --force \
    --min-hits-truth 7 \
    --min-hits-reco 4 \
    --min-pt 0. \
    --frac-reco-matched 0.5 \
    --frac-truth-matched 0.5
