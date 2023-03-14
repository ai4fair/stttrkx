## Track Reconstruction and Evaluation

This folder contains **evaluation** of reconstructed tracks.

&nbsp;

After pipeline is finished, all we get is the `edge_score`. One needs to use this information to build tracks. All post-training hepler code is located in the `eval/` directory.

&nbsp;

The track building and track evaluation is performed on the data from the GNN stage (last stage in the pipeline). Follow the procedure here,

```bash
# gnn4itk scripts
$ ./trkx_from_gnn.sh         # runs DBSCAN to build tracks
$ ./eval_reco_trkx.sh        # runs two-way matching to evaluate tracks
$ ./plot_trk_perf.sh         # plots pt, d0, theta, phi agains efficiencies
```


```bash
# exatrkx-hsf scripts
$ ./hsf_trkx_building.sh     # runs CCL to build tracks
$ ./hsf_trkx_eval.sh         # runs ATLAS, one-way and two-way matching to evaluate tracks
$ ./plot_trk_perf.sh         # plots pt, d0, theta, phi agains efficiencies
```

```bash
# gnn4itk_cf scripts
$ track_building stage       # run CCL, Evaluation and Plotting
```