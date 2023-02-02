#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from ..dnn_base import DNNBase
from ..utils.dnn_utils import make_mlp


# Edge Classification with LayerNorm
class EdgeClassifier(DNNBase):
    """A Dense Network for Edge Classification. Norm Options: Layer or Batch Norms."""
    
    def __init__(self, hparams):
        super().__init__(hparams)
        
        # Input Size: 2*(Node Features)
        input_dim = (hparams["spatial_channels"]+hparams["cell_channels"])*2
        layer_dim = [1000,2000,2000,2000,1000,1]
        # Create a Dense Network
        self.dense = make_mlp(
            input_size=input_dim,                            # Features
            sizes=layer_dim,                                 # Nodes
            hidden_activation=hparams["hidden_activation"],  # Relu
            output_activation=None,                          # None
            layer_norm=hparams["layernorm"],                 # LayerNorm
            batch_norm=hparams["batchnorm"]                  # BatchNorm
            )


    def forward(self, x, edge_index):

        # split edge_index
        start, end = edge_index
        
        # get edge features
        edge_inputs = torch.cat([x[start], x[end]], dim=1)
        
        return self.dense(edge_inputs)
