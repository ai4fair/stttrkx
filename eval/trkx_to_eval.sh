#!/bin/bash

# Combined run of trkx_from_gnn.py (trkx_from_gnn.sh)
# and eval_reco_trkx.py (eval_reco_trkx.sh) scripts.


# params
maxevts=20000
epsilon=0.15
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





# Stage [dnn, gnn, agnn,...]
ann=gnn


# Data Directories
inputdir="../run_all/fwp_"$ann"_processed/pred"     # input from gnn_processed/test or pred/
outputdir="../run_all/fwp_"$ann"_segmenting/seg"    # output of trkx_from_gnn.sh to gnn_segmenting/seg
mkdir -p $outputdir

# Tracks from GNN
python trkx_from_gnn.py \
    --input-dir $inputdir \
    --output-dir $outputdir \
    --max-evts $maxevts \
    --num-workers 8 \
    --score-name "scores" \
    --edge-score-cut $edge_score_cut \
    --epsilon $epsilon \
    --min-samples 2



# fractions
fraction=0.75

# Data Directories
raw_inputdir=$inputdir
rec_inputdir=$outputdir
outputdir="../run_all/fwp_"$ann"_segmenting/eval"  # output of eval_reco_trkx.sh
outfile=$outputdir"/$fraction"
mkdir -p $outputdir


# Don't move above outfile, name will be messed up.
if (( $(echo "$fraction == 0.5" | bc -l) )); then
  fraction=$(echo "$fraction + 0.00001" | bc -l)
fi

# Evaluate Reco. Tracks
python eval_reco_trkx.py \
    --csv-path $raw_inputdir \
    --reco-track-path $rec_inputdir \
    --outname $outfile \
    --max-evts $maxevts \
    --num-workers 8 \
    --force \
    --min-pt 0.0 \
    --min-hits-truth 7 \
    --min-hits-reco 6 \
    --frac-reco-matched $fraction \
    --frac-truth-matched $fraction
