## _GNN Models_

All models are using the Message Passing [arXiv:1704.01212](https://arxiv.org/abs/1704.01212) framework with or without Residual Connections. Following models has been tested with the `GNNBase` class that provides **bidirectional** input graphs on the fly (see `handle_directed()` in `GNNBase`). In these models the aggregate fucntion for _message-passing_ has been slightly modified.


In the long run, I would like to keep models IGNN, AGNN and GCN with MPNN, Residuals and Checkpointing.

### (1) - Interaction GNN

The model `InteractionGNN` is the result of [arXiv:1612.00222](https://arxiv.org/abs/1612.00222) that was adapted for the purpose of particle tracking by Exa.TrkX [arXiv:2103.06995](https://arxiv.org/abs/2103.06995). Model definition can be found in `interaction_gnn.py`. This model is already works perfectly with **bidirectional** input graphs.

- Single Aggregation

```bash
# message-passing (sum/mean/max aggregation) bidirected edges
edge_messages = scatter_add(e, end, dim=0, dim_size=x.shape[0])
edge_messages = scatter_mean(e, end, dim=0, dim_size=x.shape[0])
edge_messages = scatter_max(e, end, dim=0, dim_size=x.shape[0])[0]
```

- Double Aggregation

```bash
# message-passing (sum_max aggregation) bidirected edges
edge_messages = torch.cat(
    [
        scatter_max(e, end, dim=0, dim_size=x.shape[0])[0],
        scatter_add(e, end, dim=0, dim_size=x.shape[0]),
    ],
    dim=-1,
)
```

```bash
# message-passing (mean_sum aggregation) bidirected edges
edge_messages = torch.cat(
    [
        scatter_mean(e, end, dim=0, dim_size=x.shape[0]),
        scatter_add(e, end, dim=0, dim_size=x.shape[0]),
    ],
    dim=-1,
)
```

```bash
# message-passing (mean_max aggregation) bidirected edges
edge_messages = torch.cat(
    [
        scatter_max(e, end, dim=0, dim_size=x.shape[0])[0],
        scatter_mean(e, end, dim=0, dim_size=x.shape[0]),
    ],
    dim=-1,
)
```

We can have same aggregation functions for AGNN, GCN, etc. In order to compare all together, make sure the all use same aggregation functions.

### (2) - Attention GNN (AGNN)

The model `VanillaAGNN` is the attention model without a residual (aka "skip") connection.
It is the new implimentation of the `GNNSegmentClassifier` model presented in CTD2018 by 
Steven Farrell. Model definition can be found in `vanilla_agnn.py`. 

The model `ResAGNN` is the attention model with a residual (aka "skip") connection. It was tested in 
_Performance of a geometric deep learning pipeline for HL-LHC particle tracking [arXiv:2103.06995](https://arxiv.org/abs/2103.06995) by Exa.TrkX. Model definition can be found in `residual_agnn.py`. 


**NB.** AGNN is **probably** based on GAT from [arXiv:1710.10903](https://arxiv.org/abs/1710.10903). Verify?


```bash
# message-passing (sum aggregation) directed edges
edge_messages = scatter_add(
    e[:, None] * x[start], end, dim=0, dim_size=x.shape[0]
) + scatter_add(
    # e[:, None] * x[start], end, dim=0, dim_size=x.shape[0]     # gave an error
    e[:, None] * x[end], start, dim=0, dim_size=x.shape[0]
)
```

```bash
# message-passing (sum aggregation) bidirected edges
edge_messages = scatter_add(
    e * x[start], end, dim=0, dim_size=x.shape[0]
)
```

### (3) - Graph Convolutional Network (GCN)

The model `VanillaGCN` proposed by Thomas Kipf in paper [arXiv:1609.02907](https://arxiv.org/abs/1609.02907). Model definition can be found in `vanilla_gcn.py`. Model with residual connection, `ResGCN` can be found in `residual_gcn.py`. The message-passing (aggregation function) for **bidirectional** input graphs is modified as follows:


```bash
# message-passing (sum aggregation) directed edges
messages = scatter_add(
    x[start], end, dim=0, dim_size=x.shape[0]
) + scatter_add(
    x[end], start, dim=0, dim_size=x.shape[0]
)
```

```bash
# message-passing (sum aggregation) bidirected edges
messages = scatter_add(
    x[start], end, dim=0, dim_size=x.shape[0]
)
```


## Miscellaneous

VanillaAGNN is the attention model without a residual (aka "skip") connection, while ResAGNN has a skip connection. I would also point out one important but subtle difference between these two models and the InteractionNetwork implementation. 

The [aggregation operation](https://github.com/HSF-reco-and-software-triggers/Tracking-ML-Exa.TrkX/blob/89cc6090cf300591e78b6e62d15f28e3bca3987b/Pipelines/TrackML_Example/LightningModules/GNN/Models/vanilla_agnn.py#L66) can be implemented in three ways:

1. It can take `edges` going in both directions and `sum` them. This is what the `AGNN` and `ResAGNN` models do. That is, they assume there is only one edge pointing in some unimportant way between nodes.
2. It can take `edges` going in a single direction and `sum` them. The `InteractionNetwork` [operation](https://github.com/HSF-reco-and-software-triggers/Tracking-ML-Exa.TrkX/blob/89cc6090cf300591e78b6e62d15f28e3bca3987b/Pipelines/TrackML_Example/LightningModules/GNN/Models/interaction_gnn.py#L84) does this. So one might need to double the number of edges by `flipping` them, as is done in the `handle_directed()` function in GNNBase,
3. It can take `edges` going in both directions and handle them separately, as the [original version](https://github.com/HSF-reco-and-software-triggers/Tracking-ML-Exa.TrkX/blob/89cc6090cf300591e78b6e62d15f28e3bca3987b/Pipelines/TrackML_Example/LightningModules/GNN/Models/agnn.py#L78) of the `agnn` did. This might improve performance, but I haven't studied it carefully.

This is a little tricky, so let me know if it doesn't make sense. In short: Make sure information can pass in both directions from node to node. How exactly you do that (whether you make edges go in both directions, or you make the aggregation go in both directions), is not really important.

And, just to zoom out, I would say: **You probably won't see significant performance differences between the Interaction Network with a skip connection, and the AGNN with a skip connection**. They have a similar level of expressiveness. What might improve performance is in the choice of **aggregation function**, or **increasing the dataset size** and **model size**.
