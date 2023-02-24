"""
This file contains some common helper code for the analysis notebooks.
"""

# System
import os
import yaml
import pickle
from collections import namedtuple

# Externals
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
import torch
from torch.utils.data import Subset, DataLoader


# Define our Metrics class as a namedtuple
Metrics = namedtuple('Metrics', ['accuracy', 'precision', 'recall', 'f1',
                                 'prc_precision', 'prc_recall', 'prc_thresh',
                                 'roc_fpr', 'roc_tpr', 'roc_thresh', 'roc_auc'])

def compute_metrics(preds, targets, threshold=0.5):
    """Input preds, target as Numpy"""    
    # Decision boundary metrics
    y_pred, y_true = (preds > threshold), (targets > threshold)
    
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)    
    # precision = sklearn.metrics.precision_score(y_true, y_pred)
    # recall = sklearn.metrics.recall_score(y_true, y_pred)
    precision, recall, f1, support = sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    # Precision-Recall Curve
    prc_precision, prc_recall, prc_thresh = sklearn.metrics.precision_recall_curve(y_true, preds)
    
    # ROC Curve
    roc_fpr, roc_tpr, roc_thresh = sklearn.metrics.roc_curve(y_true, preds)
    roc_auc = sklearn.metrics.auc(roc_fpr, roc_tpr)
    
    # Organize metrics into a namedtuple
    metrics = Metrics(accuracy=accuracy, precision=precision, recall=recall, f1=f1,
                   prc_precision=prc_precision, prc_recall=prc_recall, prc_thresh=prc_thresh,
                   roc_fpr=roc_fpr, roc_tpr=roc_tpr, roc_thresh=roc_thresh, roc_auc=roc_auc)
    return metrics


def plot_metrics(preds, targets, metrics, name="gnn", scale='linear'):
    # Prepare the values
    labels = targets > 0.5

    # Create the Figure
    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(16,5))

    # Plot Model Outputs
    binning=dict(bins=25, range=(0,1), histtype='step', log=True)
    ax0.hist(preds[labels==True], label='Real Edges', **binning)
    ax0.hist(preds[labels==False], label='Fake Edges', **binning)
    ax0.set_xlabel('Model Output', size=14)
    ax0.legend(loc=0)

    # Plot Precision & Recall
    ax1.plot(metrics.prc_thresh, metrics.prc_precision[:-1], label='Edge Purity')  # Purity
    ax1.plot(metrics.prc_thresh, metrics.prc_recall[:-1], label='Edge Efficiency') # Efficiency
    ax1.set_xlabel('Edge Score Cut', size=14) # Model Threshold
    ax1.legend(loc=0)

    # Plot the ROC curve
    ax2.plot(metrics.roc_fpr, metrics.roc_tpr, color="darkorange", label="AUC = %.5f" % metrics.roc_auc)
    ax2.plot([0, 1], [0, 1], color="navy", linestyle="--")
    ax2.set_xlabel('False Positive Rate', size=14)
    ax2.set_ylabel('True Positive Rate', size=14)
    ax2.set_xscale(scale)
    ax2.legend(loc=0)
    
    fig.tight_layout()
    fig.savefig(name+"_metrics.pdf")

    
def plot_outputs_roc(preds, targets, metrics, name="gnn"):
    # Prepare the values
    labels = targets > 0.5

    # Figure & Axes
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12,5))

    # Plot Model Outputs
    binning=dict(bins=25, range=(0,1), histtype='step', log=True)
    ax0.hist(preds[labels==True], label='True Edges', **binning)    # True Edges
    ax0.hist(preds[labels==False], label='False Edges', **binning)  # False Edges
    
    # Axes Params
    ax0.set_xlabel('Model Output', size=16)
    ax0.set_ylabel('Counts', size=16)
    ax0.tick_params(axis='both', which='major', labelsize=12)
    ax0.tick_params(axis='both', which='minor', labelsize=12)
    # ax0.set_ylim(ymin=0.005)
    ax0.legend(fontsize=14, loc='best')


    # Plot ROC Curve with AUC
    ax1.plot(metrics.roc_fpr, metrics.roc_tpr, color="darkorange", label="AUC = %.5f" % metrics.roc_auc)
    ax1.plot([0, 1], [0, 1], color="navy", linestyle="--")
    
    # Axes Params
    ax1.set_xlabel('False Positive Rate', size=16)
    ax1.set_ylabel('True Positive Rate', size=16)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.tick_params(axis='both', which='minor', labelsize=12)
    ax1.legend(fontsize=14, loc='best')
    
    # Fig Params
    fig.tight_layout()
    fig.savefig(name+"_outputs_roc.pdf")
 
 
def plot_model_output(preds, targets, name="gnn"):
    # Prepare the values
    labels = targets > 0.5

    # Figure & Axes
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,7))

    # Ploting
    binning=dict(bins=25, range=(0,1), histtype='step', log=True)
    ax.hist(preds[labels==True], label='True Edges', **binning)    # True Edges
    ax.hist(preds[labels==False], label='False Edges', **binning)  # False Edges
    
    # Axes Params
    ax.set_xlabel('Model Output', size=20)
    ax.set_ylabel('Counts', size=20)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    # ax.set_ylim(ymin=.005)
    ax.legend(fontsize=14, loc='best')
    
    # Fig Params
    fig.tight_layout()
    fig.savefig(name+"_outputs.pdf")
    

