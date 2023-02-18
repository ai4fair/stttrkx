#!/usr/bin/env python
# coding: utf-8
import sys, os, glob, yaml
import math
import random
import numpy as np
import pandas as pd
import trackml.dataset
import seaborn as sns
import torch
import ROOT

sys.path.append('src')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
input_dir='./data_sets/ctd2022/train_100k/'
all_files = os.listdir(input_dir)
file_prefixes = sorted(os.path.join(input_dir, f.replace('-hits.csv', '')) for f in all_files if f.endswith('-hits.csv'))

inputdir='./data_sets/ctd2022/run_100k/feature_store'
proc_files = sorted(glob.glob(os.path.join(inputdir, "*")))

print("file_prefixes: ", len(file_prefixes))
print("proc_files   : ", len(proc_files))

hc1 = ROOT.TH1F("hc1", "Number of Hits;No. of Hits;Events", 200, 100, 500)
hc2 = ROOT.TH1F("hc2", "Number of Hits;No. of Hits;Events", 200, 100, 500)
hc3 = ROOT.TH1F("hc3", "Number of Hits;No. of Hits;Events", 200, 100, 500)

i = 0
t = 0

print("Filling Histograms...")

for f in proc_files[:10]:
    if i != 0 and i%1000 == 0:
        print("Processed Events:", i)
    
    print("file", f)
    
    # hits = trackml.dataset.load_event_hits(f)
    # nhits = hits.hit_id.count()
    feature_data = torch.load(f, map_location=device)
    
    # get true_edges
    e = feature_data.edge_index
    pid = feature_data.pid
    true_edges = pid[e[0]] == pid[e[1]]
    false_edges = e[:, ~true_edges]
    
    print("edge_index : ", e.size())
    print("true_edges : ", true_edges.size())
    print("false_edges: ", false_edges.size())
    
    
    #hc1.Fill(nhits)
    #hc2.Fill(nhits*100)
    #hc3.Fill(nhits*1000)
    
    i = i+1
    t = t+nhits


print("Total number of hits: ", t)

can = ROOT.TCanvas("can","Histograms",600,500)
can.cd();hc1.GetXaxis().SetTitleOffset(1.5);hc1.Draw();hc2.Draw("SAME");hc3.Draw("SAME");can.Draw();can.SaveAs("pair.pdf");can.SaveAs("pair.C");can.Close()


