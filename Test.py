#!/usr/bin/env python
# coding: utf-8

import glob, os, sys, yaml
import argparse
import logging

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import pprint
pp = pprint.PrettyPrinter(indent=2)
import seaborn as sns
import trackml.dataset
import torch
from torch_geometric.data import Data
import itertools
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from LightningModules.GNN import InteractionGNN
from LightningModules.GNN import InteractionGNN
from LightningModules.GNN import GNNBuilder, GNNMetrics
from LightningModules.GNN.Models.infer import GNNTelemetry
from LightningModules.GNN.utils.data_utils import split_datasets, load_dataset


class SttDataModule(pl.LightningDataModule):
    """"DataModules are a way of decoupling data-related hooks from the LightningModule"""
    def __init__(self, hparams):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters(hparams)
        
        # Set workers from hparams
        self.n_workers = (
            self.hparams["n_workers"]
            if "n_workers" in self.hparams
            else len(os.sched_getaffinity(0))
        )
        
        self.data_split = (
            self.hparams["train_split"]
            if "train_split" in self.hparams
            else [0,0,5000]
        )
        
        self.trainset, self.valset, self.testset = None, None, None
        self.predset = None
        
        
    def print_params(self):
        pp.pprint(self.hparams)
        
    def setup(self, stage=None):
        
        if stage == "fit" or stage is None:
            self.trainset, self.valset, self.testset = split_datasets(**self.hparams)

        if stage == "test" or stage is None:
            print("Number of Test Events: ", self.hparams['train_split'][2])
            self.testset = load_dataset(self.hparams["input_dir"], self.data_split[2])
            
        if stage == "pred" or stage is None:
            print("Number of Pred Events: ", self.hparams['train_split'][2])
            self.predset = load_dataset(self.hparams["input_dir"], self.data_split[2])

    def train_dataloader(self):
        if self.trainset is not None:
            return DataLoader(
                self.trainset, batch_size=1, num_workers=self.n_workers
            )  # , pin_memory=True, persistent_workers=True)
        else:
            return None

    def val_dataloader(self):
        if self.valset is not None:
            return DataLoader(
                self.valset, batch_size=1, num_workers=self.n_workers
            )  # , pin_memory=True, persistent_workers=True)
        else:
            return None

    def test_dataloader(self):
        if self.testset is not None:
            return DataLoader(
                self.testset, batch_size=1, num_workers=self.n_workers
            )  # , pin_memory=True, persistent_workers=True)
        else:
            return None

    #def predict_dataloader(self):
    #    if self.predset is not None:
    #        return DataLoader(
    #            self.predset, batch_size=1, num_workers=self.n_workers
    #        )  # , pin_memory=True, persistent_workers=True)
    #    else:
    #        return None


# 1 - Helper Function
def get_input_data(batch):
    input_data = batch.x
    input_data[input_data != input_data] = 0

    return input_data


# 2 - Helper Function
def handle_directed(batch, edge_sample, truth_sample, directed=False):

    edge_sample = torch.cat([edge_sample, edge_sample.flip(0)], dim=-1)
    truth_sample = truth_sample.repeat(2)

    if directed:
        direction_mask = batch.x[edge_sample[0], 0] < batch.x[edge_sample[1], 0]
        edge_sample = edge_sample[:, direction_mask]
        truth_sample = truth_sample[direction_mask]

    return edge_sample, truth_sample

def main ():
    
    # Load Model Checkpoint
    ckpnt_path = "run_all/lightning_models/lightning_checkpoints/GNNStudy/version_1/checkpoints/last.ckpt"
    checkpoint = torch.load(ckpnt_path, map_location=device)
    pp.pprint(checkpoint.keys())
    
    # View Hyperparameters
    hparams = checkpoint["hyper_parameters"]
    
    # One Can Modify Hyperparameters
    hparams["checkpoint_path"] = ckpnt_path
    hparams["input_dir"] = "run/feature_store"
    hparams["output_dir"] = "run/gnn_processed"
    hparams["artifact_library"] = "lightning_models/lightning_checkpoints"
    hparams["train_split"] = [0, 0, 10000]
    hparams["map_location"] = device

    # Init InteractionGNN
    model = InteractionGNN(hparams)
    model = model.load_from_checkpoint(**hparams)

    # Init DataModule
    dm = SttDataModule(hparams)
    dm.setup(stage="test")

    # get test_dataloader
    test_dataloader = dm.test_dataloader()

    # prepare model input
    batch = next(iter(test_dataloader))
    truth = batch.y_pid
    edge_sample, truth_sample = handle_directed(batch, batch.edge_index, truth)
    input_data = get_input_data(batch)

    # Predict
    model.eval();
    
    with torch.no_grad():
        output = model(input_data, edge_sample).squeeze()
        score = torch.sigmoid(output)
        batch.scores = score
   
   
   print("Prediction Finished")
   print("\nScores: ", batch.scores) 
    
if __name__ == "__main__":
    main()

