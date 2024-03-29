#!/bin/bash

# This script runs 'plot_trk_perf.py'

# Stage [dnn, gnn, agnn,...]
ann=gnn

infile="../run_all/fwp_"$ann"_segmenting/eval/all_particles.h5"
outputdir="../run_all/fwp_"$ann"_segmenting/eval"
prefix="trk_perf"

# reco tracks from GNN
python plot_trk_perf.py -f $infile -o $outputdir -p $prefix
