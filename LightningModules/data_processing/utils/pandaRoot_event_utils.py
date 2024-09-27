import os
import logging
import torch
import numpy as np
import pandas as pd

from typing import Tuple

from torch_geometric.data import Data
from .heuristic_utils import get_layerwise_graph, get_all_edges, graph_intersection
from .event_utils import (
    get_layerwise_edges,
    get_modulewise_edges,
    get_orderwise_edges,
    get_time_ordered_true_edges,
)
from .root_file_reader import ROOTFileReader

def prepare_sim_event():
    print("Preparing simulated event")

def prepare_digi_event():
    print("Preparing digi event")