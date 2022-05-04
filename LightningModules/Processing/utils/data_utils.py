#!/usr/bin/env python
# coding: utf-8

"""Reader Class for the PyTorch Geometric Data Produced by Processing Stage."""

import os
import glob
import torch
from typing import Any


# PyTorch Geometric Data Reader
class SttTorchDataReader(object):
    """Torch Geometric Data Reader from an Input Directory."""
    
    def __init__(self, input_dir: str):
        """Initialize Instance Variables in Constructor"""
        
        self.path = input_dir
        all_files = sorted(glob.glob(os.path.join(input_dir, "*")))
        self.nevts = len(all_files)
        self.all_evtids = [os.path.basename(x) for x in all_files]
    
    def read(self, evtid: int = None):
        """Read an Event from the Input Directory."""
        event_fname = os.path.join(self.path, "{}".format(evtid))
        event = torch.load(event_fname)
        return event
    
    def __call__(self, evtid: int, *args: Any, **kwds: Any) -> Any:
        return self.read(evtid)
