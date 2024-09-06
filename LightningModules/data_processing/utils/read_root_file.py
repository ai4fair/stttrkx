import uproot as up
import pandas as pd

defaultHitTreeNames   = ["hits", "cells"]
defaultTruthTreeNames = ["particles", "truth"]

def load_event(inputRootFile, eventNum : int, readTruth : bool, hitTreeNames : list = defaultHitTreeNames, truthTreeNames : list = defaultTruthTreeNames):
    
    if readTruth:
        return tuple(save_tree_as_pandas(rootFile = inputRootFile, treeName = treeName, eventNum = eventNum) for treeName in hitTreeNames + truthTreeNames)
    else:
        return tuple(save_tree_as_pandas(rootFile = inputRootFile, treeName = treeName, eventNum = eventNum) for treeName in hitTreeNames)


def save_tree_as_pandas(rootFile : up.ReadOnlyDirectory, treeName : str, eventNum : int):
    
    branches = rootFile[treeName].keys()
    treeData = {}

    for branch in branches:
        branchData = rootFile[treeName + "/" + branch].array(entry_start=eventNum, entry_stop=eventNum+1)
        treeData[branch] = branchData[0]

    return pd.DataFrame(treeData)
