#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from ..dnn_base import DNNBase
from ..utils.dnn_utils import make_mlp


# Edge Classification
class EdgeClassifier(DNNBase):
    """A Deep Neural Network for Edge Classification"""
    
    def __init__(self, hparams):
        super().__init__(hparams)
        
        # Input Size: 2*(Node Features)
        input_dim = (hparams["spatial_channels"] + hparams["cell_channels"])*2
        
        # Create a Dense Network for Edge Classification
        self.network = make_mlp(input_size=input_dim,
                                sizes=[1000,2000,2000,2000,1000,1],
                                hidden_activation="ReLU",
                                output_activation=None,
                                layer_norm=False
                                )
    
    
    def forward(self, x, edge_index):
    
        # indices (node_1 + node_2)
        start, end = edge_index
        
        # edges (node_1 + node_2) 
        edge_inputs = torch.cat([x[start], x[end]], dim=1)
        
        return self.network(edge_inputs)

