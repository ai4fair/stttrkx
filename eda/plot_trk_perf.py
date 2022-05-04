#!/usr/bin/env python

"""Plot track efficiency and purity"""

import os
import numpy as np
import pandas as pd
from functools import partial
from utils_plot import make_cmp_plot, pt_configs

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot tracking performance")
    add_arg = parser.add_argument
    add_arg("-f", "--fname", help="input file from eval_rec_trkx", required=True)
    add_arg("-o", "--outdir", help='output directory that saves outputs', required=True)
    add_arg('-p', '--prefix', default='out', help='prefix of output names')

    args = parser.parse_args()
    if not os.path.exists(args.fname):
        print("{} does not exists".format(args.fname))
        exit(1)

    outdir = args.outdir
    os.makedirs(os.path.abspath(outdir), exist_ok=True)
    out_prefix = args.prefix

    with pd.HDFStore(args.fname, 'r') as f:
        df = f['data']

    pt = df.pt.values
    eta = df.eta.values
    vx = df.vx.values
    vy = df.vy.values
    d0 = np.sqrt(vx ** 2 + vy ** 2)
    z0 = df.vz.values

    rectable_idx = df.is_trackable
    matched_idx = df.is_matched

    # plot the efficiency as a function of pT, eta
    make_cmp_plot_fn = partial(make_cmp_plot,
                               legends=["Generated", "Reconstructable", "Matched"],
                               ylabel="Events", ratio_label='Track efficiency',
                               ratio_legends=["Physics Eff", "Technical Eff"])

    # fiducial cuts: pT > 1 GeV and |eta| < 4
    """
    all_cuts = [(1, 4), (0.5, 4)]
    for (cut_pt, cut_eta) in all_cuts:
        cuts = (pt > cut_pt) & (np.abs(eta) < cut_eta)
        gen_pt = pt[cuts]
        true_pt = pt[cuts & rectable_idx]
        reco_pt = pt[cuts & rectable_idx & matched_idx]
        make_cmp_plot_fn([gen_pt, true_pt, reco_pt], 
            configs=pt_configs, xlabel="pT [GeV]",
            outname=os.path.join(outdir, "{}_pt_cut{}_{}".format(out_prefix, cut_pt*1000, cut_eta)),
            ymin=0.6)

        gen_eta = eta[cuts]
        true_eta = eta[cuts & rectable_idx]
        reco_eta = eta[cuts & rectable_idx & matched_idx]
        make_cmp_plot_fn([gen_eta, true_eta, reco_eta], configs=eta_configs, xlabel=r"$eta$",
            outname=os.path.join(outdir, "{}_eta_cut{}_{}".format(out_prefix, cut_pt*1000, cut_eta)),
            ymin=0.6)
    """
