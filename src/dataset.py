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

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger 

# Ray Tune
from ray import air, tune
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback

from LightningModules.DNN.Models.tune_network import EdgeClassifier


class SttDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        
        
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
        
        
    def pparams (self):
        pp.pprint(self.hparams)
        
    def setup(self, stage=None):
        
        if stage == "fit" or stage is None:
            self.trainset, self.valset, self.testset = split_datasets(**self.hparams)

        if stage == "test" or stage is None:
            print("Number of Test Events: ", self.hparams['train_split'][2])
            self.testset = load_dataset(self.hparams["input_dir"], self.data_split[2])
            
        if stage == "pred":
            self.predset = None

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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("3_Train_GNN.py")
    add_arg = parser.add_argument
    add_arg("config", nargs="?", default="pipeline_config.yaml")
    return parser.parse_args()


def train (config_file="pipeline_config.yaml"):

    logging.info(headline("Step 3: Running GNN training "))

    with open(config_file) as file:
        all_configs = yaml.load(file, Loader=yaml.FullLoader)
    
    common_configs = all_configs["common_configs"]
    dnn_configs = all_configs["model_configs"]

    logging.info(headline("a) Initialising model" ))

    model = EdgeClassifier(dnn_configs)
    print(model)

    logging.info(headline("b) Running training" ))

    save_directory = common_configs["artifact_directory"]
    logger = TensorBoardLogger(save_directory, name=common_configs["experiment_name"], version=None)

    trainer = pl.Trainer(
        gpus=common_configs["gpus"],
        max_epochs=1, # dnn_configs["max_epochs"],
        logger=logger
    )

    trainer.fit(model)

    logging.info(headline("c) Saving model" ))
    
    os.makedirs(save_directory, exist_ok=True)
    trainer.save_checkpoint(os.path.join(save_directory, common_configs["experiment_name"]+".ckpt"))

    return trainer, model


if __name__ == "__main__":

    args = parse_args()
    config_file = args.config
    train(config_file)

