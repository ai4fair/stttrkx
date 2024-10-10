import os
import logging
import torch
import numpy as np
import pandas as pd

from torch_geometric.data import Data
from .heuristic_utils import get_layerwise_graph, get_all_edges, graph_intersection
from .event_utils import (
    get_layerwise_edges,
    get_modulewise_edges,
    get_orderwise_edges,
    get_time_ordered_true_edges,
)
from .particle_utils import is_signal_particle, get_process_ids, get_all_mother_ids

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

    event_id = event["event_id"]

    # Prepare the output filename and check if it already exists
    output_filename = f"{output_dir}/event_{event_id}.pt"
    if not os.path.exists(output_filename) or overwrite:
        logging.info(f"Writing into {output_filename}")
    else:
        logging.warning(
            f"File {output_filename} already exists! Skipping event {event_id}..."
        )
        return

    # Get the mother ids of all particles.
    mother_ids = get_all_mother_ids(
        mother_ids=event["mother_id"],
        second_mother_ids=event["second_mother_id"],
    )

    # Create a dictionary to store the processed mcTrack information.
    mcTrack_dict = {}

    # Get the track ids of particles that leave a signal in the STT
    # and save them into the dictionary.
    unique_track_ids = np.unique(np.array(event["particle_id"]))
    mcTrack_dict["particle_id"] = unique_track_ids

    # Initialize the "is_signal" column of the dictionary with an empty bool array.
    mcTrack_dict["primary"] = np.empty(len(unique_track_ids), dtype=bool)

    # mcTrack keys that should be saved and processed.
    mcTrack_keys = [
        "vx",
        "vy",
        "vz",
        "pdgcode",
    ]

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
            process_ids=event["process_code"],
            mother_ids=mother_ids,
            pdg_ids=event["pdgcode"],
            particle_id=particle_id,
        )
        # Check if the particle is a signal particle.
        mcTrack_dict["primary"][particle_num] = is_signal_particle(
            process_mc_ids=mc_ids,
            process_ids=process_codes,
            signal_mc_ids=signal_signatures["particle_ids"],
            signal_process_ids=signal_signatures["process_codes"],
        )
        particle_num += 1

    # Create a pandas DataFrame from the mcTrack dictionary.
    processed_df = pd.DataFrame(mcTrack_dict)
    del mcTrack_dict

    sttP_dict = {}

    for key in key_dict["sttPoint"]:
        sttP_dict[key] = event[key]

    sttP_dict["hit_id"] = np.arange(len(sttP_dict[key_dict["sttPoint"][0]]))

    # Calculate the transverse momentum.
    sttP_dict["ppt"] = np.sqrt(sttP_dict["tpx"] ** 2 + sttP_dict["tpy"] ** 2)
    # Calculate the polar angle theta.
    sttP_dict["ptheta"] = np.arctan2(sttP_dict["ppt"], sttP_dict["tpz"])
    # Calculate the azimuthal angle phi.
    sttP_dict["pphi"] = np.arctan2(sttP_dict["tpy"], sttP_dict["tpx"])
    # Calculate the pseudorapidity eta.
    sttP_dict["peta"] = -np.log(np.tan(sttP_dict["ptheta"] / 2.0))

    processed_df = pd.merge(pd.DataFrame(sttP_dict), processed_df, on="particle_id")
    del sttP_dict

    sttH_dict = {}

    for key in key_dict["sttHit"]:
        sttH_dict[key] = event[key]

    sttH_dict["layer_id"] = np.empty(len(sttH_dict[key_dict["sttHit"][0]]), dtype=int)
    sttH_dict["sector_id"] = np.empty(len(sttH_dict[key_dict["sttHit"][0]]), dtype=int)
    sttH_dict["skewed"] = np.empty(len(sttH_dict[key_dict["sttHit"][0]]), dtype=int)

    hit_num = 0
    for tube_id in sttH_dict["module_id"]:
        sttH_dict["layer_id"][hit_num] = stt_geo["layerID"][tube_id - 1]
        sttH_dict["sector_id"][hit_num] = stt_geo["sectorID"][tube_id - 1]
        sttH_dict["skewed"][hit_num] = stt_geo["skewed"][tube_id - 1]
        hit_num += 1

    # Calculate the transverse distance (r), azimuthal angle (phi), polar angle (theta), and pseudo-rapidity (eta)
    sttH_dict["r"] = np.sqrt(
        sttH_dict["x"] ** 2 + sttH_dict["y"] ** 2
    )  # Transverse distance from the interaction point
    sttH_dict["phi"] = np.arctan2(sttH_dict["y"], sttH_dict["x"])  # Azimuthal angle
    sttH_dict["theta"] = np.arccos(
        sttH_dict["z"]
        / np.sqrt(sttH_dict["x"] ** 2 + sttH_dict["y"] ** 2 + sttH_dict["z"] ** 2)
    )  # Polar angle
    sttH_dict["eta"] = -np.log(np.tan(sttH_dict["theta"] / 2.0))  # Pseudo-rapidity

    processed_df = pd.merge(pd.DataFrame(sttH_dict), processed_df, on="hit_id")

    # skip noise hits.
    if not kwargs["noise"]:
        processed_df = processed_df.query("primary==1")

    # skip skewed tubes
    if not kwargs["skewed"]:
        processed_df = processed_df.query("skewed==0")

    processed_df = processed_df.assign(event_id=event_id)

    # Get the true edges using the true time order of the hits
    true_edges = get_time_ordered_true_edges(processed_df)
    logging.info(
        f"Time ordered truth graph built for {event_id} with size {true_edges.shape}"
    )

    # Build input edges by connecting all hits to all other hits.
    input_edges = get_all_edges(processed_df)
    logging.info(f"All input graph built for {event_id} with size {input_edges.shape}")

    # feature scale for X=[r,phi,z]
    feature_scale = [100, np.pi, 100]

    # Build the PyTorch Geometric (PyG) 'Data' object
    data = Data(
        x=torch.from_numpy(
            processed_df[["r", "phi", "isochrone"]].to_numpy() / feature_scale
        ).float(),
        pid=torch.from_numpy(processed_df["particle_id"].to_numpy()),
        hid=torch.from_numpy(processed_df["hit_id"].to_numpy()),
        pt=torch.from_numpy(processed_df["ppt"].to_numpy()),
        vertex=torch.from_numpy(processed_df[["vx", "vy", "vz"]].to_numpy()),
        pdgcode=torch.from_numpy(processed_df["pdgcode"].to_numpy()),
        ptheta=torch.from_numpy(processed_df["ptheta"].to_numpy()),
        peta=torch.from_numpy(processed_df["peta"].to_numpy()),
        pphi=torch.from_numpy(processed_df["pphi"].to_numpy()),
        true_edges=torch.from_numpy(true_edges),
        primary=torch.from_numpy(processed_df["primary"].to_numpy()),
        event_file=event_id,
    )

    # Get the input and true edges as PyTorch tensors
    input_edges = torch.from_numpy(input_edges)
    true_edges = data.true_edges

    # Label the input edges, and reorganizes the order of the edges to fit the labels
    new_input_edges, y = graph_intersection(input_edges, true_edges)

    # Save both the labels and edges in the data object
    data.edge_index = new_input_edges
    data.y_pid = y

    # Save the data object to a PyTorch file
    with open(output_filename, "wb") as output_file:
        torch.save(data, output_file)
