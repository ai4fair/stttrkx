#!/bin/bash

ann=dnn

# Data Directories
raw_inputdir="../run_all/"$ann"_processed/pred"     # output of DNN stage in test/pred
echo $raw_inputdir

rec_inputdir="../run_all/"$ann"_segmenting/seg"     # output of trkx_from_gnn.sh
echo $rec_inputdir

outputdir="../run_all/"$ann"_segmenting/eval"   # output of eval_reco_trkx.sh
echo $outputdir

outfile=$outputdir"/all"
echo $outfile
