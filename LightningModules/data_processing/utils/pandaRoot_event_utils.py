import os
import logging
import torch
import numpy as np
import pandas as pd
import awkward as ak

from typing import Tuple
from pprint import pprint

from torch_geometric.data import Data
from .heuristic_utils import get_layerwise_graph, get_all_edges, graph_intersection
from .event_utils import (
    get_layerwise_edges,
    get_modulewise_edges,
    get_orderwise_edges,
    get_time_ordered_true_edges,
)
from .particle_utils import is_signal_particle, get_process_ids, get_all_mother_ids

from .root_file_reader import ROOTFileReader


def prepare_event(
    event: pd.Series,
    key_dict: dict,
    signal_signatures,
    stt_geo,
    output_dir: str,
    true_edge_method: str,
    input_edge_method: str,
    overwrite: bool,
    **kwargs,
) -> None:

    # Convert the tuple to a dictionary.
    event = event._asdict()

    # Get the mother ids of all particles.
    mother_ids = get_all_mother_ids(
        mother_ids=event["mc_mother_id"],
        second_mother_ids=event["mc_second_mother_id"],
    )

    # Create a dictionary to store the processed mcTrack information.
    mcTrack_dict = {}

    # Get the track ids of particles that leave a signal in the STT
    # and save them into the dictionary.
    unique_track_ids = np.unique(np.array(event["track_id"]))
    mcTrack_dict["track_id"] = unique_track_ids
    pprint(unique_track_ids)

    # Initialize the "is_signal" column of the dictionary with an empty bool array.
    mcTrack_dict["is_signal"] = np.empty(len(unique_track_ids), dtype=bool)

    # mcTrack keys that should be saved and processed.
    mcTrack_keys = {
        "mc_start_x",
        "mc_start_y",
        "mc_start_z",
        "mc_px",
        "mc_py",
        "mc_pz",
        "mc_pdg_code",
    }

    # Iterate over the specified and save the particle information of
    # the once leaving hits in the STT into the dictionary.
    for key in mcTrack_keys:
        mcTrack_dict[key] = event[key][unique_track_ids]

    # Iterate over all unique track ids and get the particle wise information.
    particle_num = 0
    for particle_id in unique_track_ids:
        # Get the PDG MC IDs and VMC process codes of the particle leaving the track
        # and all its mother particles.
        mc_ids, process_codes = get_process_ids(
            process_ids=event["mc_process"],
            mother_ids=mother_ids,
            pdg_ids=event["mc_pdg_code"],
            particle_id=particle_id,
        )
        # Check if the particle is a signal particle.
        mcTrack_dict["is_signal"][particle_num] = is_signal_particle(
            process_mc_ids=mc_ids,
            process_ids=process_codes,
            signal_mc_ids=signal_signatures["particle_ids"],
            signal_process_ids=signal_signatures["process_codes"],
        )
        particle_num += 1

    # Calculate the transverse momentum.
    mcTrack_dict["mc_pt"] = np.sqrt(
        mcTrack_dict["mc_px"] ** 2 + mcTrack_dict["mc_py"] ** 2
    )
    # Calculate the polar angle theta.
    mcTrack_dict["mc_theta"] = np.arctan2(mcTrack_dict["mc_pt"], mcTrack_dict["mc_pz"])
    # Calculate the azimuthal angle phi.
    mcTrack_dict["mc_phi"] = np.arctan2(mcTrack_dict["mc_py"], mcTrack_dict["mc_px"])
    # Calculate the pseudorapidity eta.
    mcTrack_dict["mc_eta"] = -np.log(np.tan(mcTrack_dict["mc_theta"] / 2.0))

    # Delete the momentum components from the dictionary.
    del mcTrack_dict["mc_px"]
    del mcTrack_dict["mc_py"]
    del mcTrack_dict["mc_pz"]

    # Create a pandas DataFrame from the mcTrack dictionary.
    processed_df = pd.DataFrame(mcTrack_dict)
    del mcTrack_dict

    sttP_dict = {}

    for key in key_dict["sttPoint"]:
        sttP_dict[key] = event[key]

    sttP_dict["hit_id"] = np.arange(len(sttP_dict[key_dict["sttPoint"][0]]))

    processed_df = pd.merge(pd.DataFrame(sttP_dict), processed_df, on="track_id")
    del sttP_dict

    sttH_dict = {}

    for key in key_dict["sttHit"]:
        sttH_dict[key] = event[key]

    sttH_dict["sttH_layer_id"] = np.empty(
        len(sttH_dict[key_dict["sttHit"][0]]), dtype=int
    )
    sttH_dict["sttH_sector_id"] = np.empty(
        len(sttH_dict[key_dict["sttHit"][0]]), dtype=int
    )
    sttH_dict["sttH_skewed"] = np.empty(
        len(sttH_dict[key_dict["sttHit"][0]]), dtype=int
    )

    hit_num = 0
    for tube_id in sttH_dict["sttH_tube_id"]:
        sttH_dict["sttH_layer_id"][hit_num] = stt_geo["layerID"][tube_id - 1]
        sttH_dict["sttH_sector_id"][hit_num] = stt_geo["sectorID"][tube_id - 1]
        sttH_dict["sttH_skewed"][hit_num] = stt_geo["skewed"][tube_id - 1]
        hit_num += 1

    # Calculate the transverse distance (r), azimuthal angle (phi), polar angle (theta), and pseudo-rapidity (eta)
    sttH_dict["sttH_r"] = np.sqrt(
        sttH_dict["sttH_x"] ** 2 + sttH_dict["sttH_y"] ** 2
    )  # Transverse distance from the interaction point
    sttH_dict["sttH_phi"] = np.arctan2(
        sttH_dict["sttH_y"], sttH_dict["sttH_x"]
    )  # Azimuthal angle
    sttH_dict["sttH_r3"] = np.sqrt(
        sttH_dict["sttH_x"] ** 2 + sttH_dict["sttH_y"] ** 2 + sttH_dict["sttH_z"] ** 2
    )  # 3D distance from the interaction point
    sttH_dict["sttH_theta"] = np.arccos(
        sttH_dict["sttH_z"] / sttH_dict["sttH_r3"]
    )  # Polar angle
    sttH_dict["sttH_eta"] = -np.log(
        np.tan(sttH_dict["sttH_theta"] / 2.0)
    )  # Pseudo-rapidity

    processed_df = pd.merge(pd.DataFrame(sttH_dict), processed_df, on="hit_id")

    # skip noise hits.
    if not kwargs["noise"]:
        processed_df = processed_df.query("primary==1")

    # skip skewed tubes
    if not kwargs["skewed"]:
        processed_df = processed_df.query("skewed==0")

    pprint(processed_df.dtypes)
    pprint(processed_df.head())