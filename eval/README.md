## Track Reconstruction and Evaluation

This folder contains **evaluation** of reconstructed tracks.

&nbsp;

After pipeline is finished, all we get is the `edge_score`. One needs to use this information to build tracks. All post-training hepler code is located in the `eval/` directory.

&nbsp;

The track building and track evaluation is performed on the data from the GNN stage (last stage in the pipeline). Follow the procedure here,

1. First, run `trkx_from_gnn.sh`
2. Second, run `trkx_reco_eval.sh`
3. Thirs, run `plot_trk_perf.sh`

