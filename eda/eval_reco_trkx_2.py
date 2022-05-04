#!/usr/bin/env python

#TODO: Script from GNN4ITK, Convert to STT case.

import os
import pandas as pd
import numpy as np

def evaluate_reco_tracks(
        truth: pd.DataFrame, reconstructed: pd.DataFrame,
        particles: pd.DataFrame,
        min_hits_truth=9, min_hits_reco=5,
        min_pt=1., frac_reco_matched=0.5, frac_truth_matched=0.5, **kwargs):
    """Return 
    
    
    Args:
        truth: a dataframe with columns of ['hit_id', 'particle_id']
        reconstructed: a dataframe with columns of ['hit_id', 'track_id']
        particles: a dataframe with columns of 
            ['particle_id', 'pt', 'eta', 'radius', 'vz'].
            where radius = sqrt(vx**2 + vy**2) and 
            ['vx', 'vy', 'vz'] are the production vertex of the particle
        min_hits_truth: minimum number of hits for truth tracks
        min_hits_reco:  minimum number of hits for reconstructed tracks

    Returns:
        A tuple of (
            n_true_tracks: int, number of true tracks
            n_reco_tracks: int, number of reconstructed tracks
            n_matched_reco_tracks: int, number of reconstructed tracks
                matched to true tracks
            matched_pids: np.narray, a list of particle IDs matched
                by reconstructed tracks
        )
    """
    # just in case particle_id == 0 included in truth.
    if 'particle_id' in truth.columns:
        truth = truth[truth.particle_id > 0]

    # get number of spacepoints in each reconstructed tracks
    n_reco_hits = reconstructed.track_id.value_counts(sort=False)\
        .reset_index().rename(
            columns={"index":"track_id", "track_id": "n_reco_hits"})

    # only tracks with a minimum number of spacepoints are considered
    n_reco_hits = n_reco_hits[n_reco_hits.n_reco_hits >= min_hits_reco]
    reconstructed = reconstructed[reconstructed.track_id.isin(n_reco_hits.track_id.values)]

    # get number of spacepoints in each particle
    hits = truth.merge(particles, on='particle_id', how='left')
    n_true_hits = hits.particle_id.value_counts(sort=False).reset_index().rename(
        columns={"index":"particle_id", "particle_id": "n_true_hits"})
    
    # only particles leaves at least min_hits_truth spacepoints 
    # and with pT >= min_pt are considered.
    particles = particles.merge(n_true_hits, on=['particle_id'], how='left')

    is_trackable = particles.n_true_hits >= min_hits_truth


    # event has 3 columnes [track_id, particle_id, hit_id]
    event = pd.merge(reconstructed, truth, on=['hit_id'], how='left')

    # n_common_hits and n_shared should be exactly the same 
    # for a specific track id and particle id

    # Each track_id will be assigned to multiple particles.
    # To determine which particle the track candidate is matched to, 
    # we use the particle id that yields a maximum value of n_common_hits / n_reco_hits,
    # which means the majority of the spacepoints associated with the reconstructed
    # track candidate comes from that true track.
    # However, the other way may not be true.
    reco_matching = event.groupby(['track_id', 'particle_id']).size()\
        .reset_index().rename(columns={0:"n_common_hits"})

    # Each particle will be assigned to multiple reconstructed tracks
    truth_matching = event.groupby(['particle_id', 'track_id']).size()\
        .reset_index().rename(columns={0:"n_shared"})

    # add number of hits to each of the maching dataframe
    reco_matching = reco_matching.merge(n_reco_hits, on=['track_id'], how='left')
    truth_matching = truth_matching.merge(n_true_hits, on=['particle_id'], how='left')

    # calculate matching fraction
    reco_matching = reco_matching.assign(
        purity_reco=np.true_divide(reco_matching.n_common_hits, reco_matching.n_reco_hits))
    truth_matching = truth_matching.assign(
        purity_true = np.true_divide(truth_matching.n_shared, truth_matching.n_true_hits))

    # select the best match
    reco_matching['purity_reco_max'] = reco_matching.groupby(
        "track_id")['purity_reco'].transform(max)
    truth_matching['purity_true_max'] = truth_matching.groupby(
        "track_id")['purity_true'].transform(max)

    matched_reco_tracks = reco_matching[
        (reco_matching.purity_reco_max >= frac_reco_matched) \
      & (reco_matching.purity_reco == reco_matching.purity_reco_max)]

    matched_true_particles = truth_matching[
        (truth_matching.purity_true_max >= frac_truth_matched) \
      & (truth_matching.purity_true == truth_matching.purity_true_max)]

    # now, let's combine the two majority criteria
    # reconstructed tracks must be in both matched dataframe
    # and the so matched particle should be the same
    # in this way, each track should be only assigned 
    combined_match = matched_true_particles.merge(
        matched_reco_tracks, on=['track_id', 'particle_id'], how='inner')

    n_reco_tracks = n_reco_hits.shape[0]
    n_true_tracks = particles.shape[0]

    # For GNN, there are non-negaliable cases where GNN-based
    # track candidates are matched to particles not considered as interesting.
    # which means there are paticles in matched_pids that do not exist in particles.
    matched_pids = np.unique(combined_match.particle_id)

    is_matched = particles.particle_id.isin(matched_pids).values
    n_matched_particles = np.sum(is_matched)

    n_matched_tracks = reco_matching[
        reco_matching.purity_reco >= frac_reco_matched].shape[0]
    n_matched_tracks_poi = reco_matching[
        (reco_matching.purity_reco >= frac_reco_matched) \
        & (reco_matching.particle_id.isin(particles.particle_id.values))
        ].shape[0]
    # print(n_matched_tracks_poi, n_matched_tracks)

    # num_particles_matched_to = reco_matched.groupby("particle_id")['track_id']\
    #     .count().reset_index().rename(columns={"track_id": "n_tracks_matched"})
    # n_duplicated_tracks = num_particles_matched_to.shape[0]
    n_duplicated_tracks = n_matched_tracks_poi - n_matched_particles

    particles = particles.assign(
        is_matched=is_matched,
        is_trackable=is_trackable)

    return (n_true_tracks, n_reco_tracks, n_matched_particles,
            n_matched_tracks, n_duplicated_tracks, n_matched_tracks_poi,
            particles)


def run_one_evt(evtid, csv_reader, recotrkx_reader, **kwargs):
    # print("Running {}".format(evtid))

    raw_data = csv_reader(evtid)
    truth = raw_data.spacepoints[['hit_id', 'particle_id']]
    particles = raw_data.particles

    submission = recotrkx_reader(evtid)
    results = evaluate_reco_tracks(truth, submission, particles, **kwargs)
    return results[:-1] + (results[-1].assign(evtid=evtid), )

# %%
if __name__ == '__main__':
    import argparse
    from multiprocessing import Pool
    import time
    from functools import partial

    from gnn4itk.io.acts_data import ACTSCSVReader
    from gnn4itk.io.athena_data import AthenaDFReader
    from gnn4itk.io.reco_trkx import RecoTrkxReader
    from gnn4itk.io.reco_trkx import ACTSTrkxReader

    parser = argparse.ArgumentParser(description='Evaluating tracking reconstruction')
    add_arg = parser.add_argument
    add_arg('-r', '--reco-track-path', help='path to reconstructed tracks', required=True)
    add_arg('-c', '--csv-path', help='path to csv path', required=True)
    add_arg('-o', '--outname', help='output name without postfix', required=True)
    add_arg('--max-evts', help='maximum number of events', type=int, default=1)
    add_arg('-e', '--event-id', help='evaluate a particular event',
        type=int, default=None)
    add_arg('-f', '--force', help='force to over write existing file', action='store_true')
    add_arg("--num-workers", help='number of workers', default=1, type=int)

    # controls which reader to use!
    add_arg('--ckf', help='tracks from ACTS CKF method', action='store_true')
    add_arg('--acts', help='use athena reader', action='store_true')
    add_arg("--df-postfix", help='post fix for CKF data from athena', default='csv')

    add_arg("--min-hits-truth", help='minimum number of hits in a truth track',
            default=7, type=int)
    add_arg("--min-hits-reco", help='minimum number of hits in a reconstructed track',
            default=4, type=int)
    add_arg('--min-pt', help='minimum pT of true track', type=float, default=1.0)
    add_arg("--frac-reco-matched", help='fraction of matched hits over total hits in a reco track',
                default=0.5, type=float)
    add_arg("--frac-truth-matched", help='fraction of matched hits over total hits in a truth track',
                default=0.5, type=float)
    
    args = parser.parse_args()
    reco_track_path = args.reco_track_path
    num_workers = args.num_workers
    outname = args.outname
    outdir = os.path.dirname(os.path.abspath(outname))
    os.makedirs(outdir, exist_ok=True)

    # read reconstructed tracks
    if args.ckf and args.acts:
        reco_trkx_reader = ACTSTrkxReader(reco_track_path)
    else:
        reco_trkx_reader = RecoTrkxReader(reco_track_path)

    n_tot_files = reco_trkx_reader.nevts
    all_evtids = reco_trkx_reader.all_evtids
    max_evts = args.max_evts if args.max_evts > 0 \
        and args.max_evts <= n_tot_files else n_tot_files

    print("Out of {} events processing {} events with {} workers".format(
        n_tot_files, max_evts, args.num_workers))
    print("Output directory:", outdir)

    # read raw CSV files to get truth information
    if args.acts:
        csv_reader = ACTSCSVReader(args.csv_path)
    else:
        csv_reader = AthenaDFReader(args.csv_path, args.df_postfix)
        

    out_array = '{}_particles.npz'.format(outname)
    if os.path.exists(out_array) and not args.force:
        print("{} is there, use -f to overwrite the file".format(out_array))
        exit(1)

    if not args.event_id:
        if num_workers > 1:
            with Pool(num_workers) as p:
                fnc = partial(run_one_evt,
                    csv_reader=csv_reader,
                    recotrkx_reader=reco_trkx_reader,
                    **vars(args))
                res = p.map(fnc, all_evtids[:max_evts])
        else:
            res = [run_one_evt(evtid, csv_reader, reco_trkx_reader, **vars(args)) \
                for evtid in all_evtids[:max_evts]]

        # merge results from each process
        n_true_tracks = sum([x[0] for x in res])
        n_reco_tracks = sum([x[1] for x in res])
        n_matched_true_tracks = sum([x[2] for x in res])
        n_matched_reco_tracks = sum([x[3] for x in res])
        n_duplicated_reco_tracks = sum([x[4] for x in res])
        n_matched_reco_tracks_poi = sum([x[5] for x in res])
        particles = pd.concat([x[-1] for x in res], axis=0)
    else:
        n_true_tracks, n_reco_tracks, 
        n_matched_true_tracks, n_matched_reco_tracks,
        n_duplicated_reco_tracks, n_matched_reco_tracks_poi, particles = \
                run_one_evt(args.event_id, csv_reader, reco_trkx_reader, **vars(args))


    with pd.HDFStore(out_array, 'w') as f:
        f['data'] = particles

    # calculate the track efficiency and purity
    out_sum = "{}_summary.txt".format(outname)
    ctime = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    summary = ["".join(['-']*50), 
        "Run Time: {:>20}".format(ctime),
        "Reconstructed tracks: {}".format(os.path.abspath(reco_track_path)),
        "# of events: {}".format(max_evts if not args.event_id else 1),
        "Truth tracks: {:>20}".format(n_true_tracks),
        "Truth tracks matched: {:>20}".format(n_matched_true_tracks),
        "Reco. tracks: {:>20}".format(n_reco_tracks),
        "Reco. tracks matched: {:>20}".format(n_matched_reco_tracks),
        "Reco. tracks matched to POI: {:>20}".format(n_matched_reco_tracks_poi),
        "Reco. tracks duplicated: {:>20}".format(n_duplicated_reco_tracks),
        "Tracking Eff.: {:>20.4f}%".format(100*n_matched_true_tracks/n_true_tracks),
        "Fake rate:     {:>20.4f}%".format(100-100*n_matched_reco_tracks/n_reco_tracks),
        "Duplication Rate: {:>20.4f}%".format(100*n_duplicated_reco_tracks/n_reco_tracks)
        ]

    with open(out_sum, 'a') as f:
        f.write("\n".join(summary))
        f.write("\n")
