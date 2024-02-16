## _Track Evaluation_

After pipeline is finished, we get _`edge scores`_ from the **DNN/GNN** stages also known as **edge labelling** or **edge classification** stages and _`track candidates`_ from the **segmenting** stage. The `eval/` directory contains code for **track evaluation** after the **segmenting**.

_**Note:**_ Due to post analysis needs of running segmenting together with track evaluation, I have put segmenting code here as a script.


The track building and track evaluation is performed on the data from the GNN stage (last stage in the pipeline). Follow the procedure here,

```bash
# gnn4itk scripts (recommended)
$ ./trkx_from_gnn.sh         # Segmenting: Track Building with DBSCAN
$ ./eval_reco_trkx.sh        # Evaluation: Track Evaluation with Two-way Matching Scheme
$ ./plot_trk_perf.sh         # Plotting: Plot track efficiencies vs pt, d0, theta, phi, etc
```

Some alternative scripts for achiving the same:

```bash
# exatrkx-hsf scripts (alternative)
$ ./hsf_trkx_building.sh     # Segmenting: Track Building with CCL
$ ./hsf_trkx_eval.sh         # Evaluation: Track Evaluation with One-ways/Two-way Matching Schemes
$ ./plot_trk_perf.sh         # Plotting: Plot track efficiencies vs pt, d0, theta, phi, etc
```