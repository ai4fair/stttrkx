## TrackML Model

### (1) - Attention GNN (AGNN)
The model `VanillaAGNN` is the attention model without a residual (aka "skip") connection.
It is the new implimentation of the `GNNSegmentClassifier` model presented in CTD2018 by 
Steven Farrell. Model definition can be found in `vanilla_agnn.py`. 


### (2) - Attention GNN with Residuals (ResAGNN)
The model `ResAGNN` is the attention model with a residual connection. It was tested in 
_Performance of a geometric deep learning pipeline for HL-LHC particle tracking [arXiv:2103.06995]_
by Exa.TrkX. Model definition can be found in `residual_agnn.py`. 


### (3) - Interaction GNN
The model `InteractionGNN` is the result of _[arXiv:1612.00222]_ that was adapted for the
purpose of particle tracking by Exa.TrkX _[arXiv:2103.06995]_. Model definition can be 
found in `interaction_gnn.py`.




## Differences
**Q**: Hi Daniel, sorry for bothering you but I have a quick and urgent question regarding 
the TrackML pipeline in Exa.Trkx repo. I have been using a customized version of it for 
PANDA detector. You probably have heard from Xiangyang. I was reading paper "Performance 
of a geometric deep learning pipeline for HL-LHC particle tracking", here there is a 
comparison made among `InteractionGNN`, `AttentionGNN` and `Residual AttentionGNN` networks.
The `AGNN` and `ResAGNN` differs only in Residual connections. What I understood from the 
paper is that:

1. AGNN is in fact the original **Hep.TrkX** model tested by Steven Farrell, it is named 
as `GNNSegmentClassifier` and was presented in CTD2018
2. I am note sure if there is published work specific to `ResAGNN` for tracking, except 
in above paper, if you know then let me know

If I look into `Pipelines/TrackML_Example/LightningModules/GNN/Models` folder, then I see 
two models `agnn.py` which is **ResAGNN** (2nd bullet) model and a `vanilla_agnn.py` which
is **AGNN** (1st. bullet) that is without residual connections.

I just need a confirmation if it is correct what I have said before I start investing time
into them. I need to run a quick test with these models for PANDA data, I already have 
result for InteractionGNN.






**ANS**: Hi Adeel! No problem about bothering me - and apologies for not replying to your 
earlier messages. We have been busy creating a new framework for GNN tracking that is
hopefully much more robust and simpler than the current HSF + TrainTrack concept. That 
is also why I have not been updating/supporting it so much in the last few months.

As for your specific question about AGNN vs ResAGNN, you're correct: VanillaAGNN is the 
attention model without a residual (aka "skip") connection, while ResAGNN has a skip 
connection. I would also point out one important but subtle difference between these two 
models and the InteractionNetwork implementation. The [aggregation operation](https://github.com/HSF-reco-and-software-triggers/Tracking-ML-Exa.TrkX/blob/89cc6090cf300591e78b6e62d15f28e3bca3987b/Pipelines/TrackML_Example/LightningModules/GNN/Models/vanilla_agnn.py#L66) can be implemented in three ways:

1. It can take `edges` going in both directions and `sum` them. This is what the `AGNN` and `ResAGNN` models do. That is, they assume there is only one edge pointing in some unimportant way between nodes.
2. It can take `edges` going in a single direction and `sum` them. The `InteractionNetwork` [operation](https://github.com/HSF-reco-and-software-triggers/Tracking-ML-Exa.TrkX/blob/89cc6090cf300591e78b6e62d15f28e3bca3987b/Pipelines/TrackML_Example/LightningModules/GNN/Models/interaction_gnn.py#L84) does this. So one might need to double the number of edges by `flipping` them, as is done in the `handle_directed()` function in GNNBase,
3. It can take `edges` going in both directions and handle them separately, as the [original version](https://github.com/HSF-reco-and-software-triggers/Tracking-ML-Exa.TrkX/blob/89cc6090cf300591e78b6e62d15f28e3bca3987b/Pipelines/TrackML_Example/LightningModules/GNN/Models/agnn.py#L78) of the `agnn` did. This might improve performance, but I haven't studied it carefully.

This is a little tricky, so let me know if it doesn't make sense. In short: Make sure information can pass in both directions from node to node. How exactly you do that (whether you make edges go in both directions, or you make the aggregation go in both directions), is not really important.

And, just to zoom out, I would say: **You probably won't see significant performance differences between the Interaction Network with a skip connection, and the AGNN with a skip connection**. They have a similar level of expressiveness. What might improve performance is in the choice of **aggregation function**, or **increasing the dataset size** and **model size**.



