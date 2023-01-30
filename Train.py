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


def headline(message):
    buffer_len = (80 - len(message))//2 if len(message) < 80 else 0
    return "-"*buffer_len + ' ' + message + ' ' + '-'*buffer_len


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

