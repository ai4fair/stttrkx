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


def select_hits(
    event_id: int, file_reader: ROOTFileReader, noise: bool, skewed: bool
) -> pd.DataFrame:
    """
    Function to select and process the parameters imported from the ROOT file
    and apply selection criteria for the hits of a single event.

    Args:
        event_id (int): Event number to be processed.
        file_reader (ROOTFileReader): ROOTFileReader object containing the open ROOT file.
        noise (bool): If True, hits from secondary processes will be filtered out.
        skewed (bool): If True, hits in the skewed tubes of the STT will be included.

    Returns:
        pd.DataFrame: _description_
    """

    # Get the pandas DataFrames for the hits, tubes, particles, and truth for the specified event using the ROOTFileReader
    hits, tubes, particles, truth = file_reader.load_event(event_id)

    # store original order (needed for orderwise_true_edges function)
    hits["original_order"] = hits.index

    # Merge the relevant columns from the particles DataFrame into the truth DataFrame
    truth = truth.merge(
        particles[["particle_id", "vx", "vy", "vz", "pdgcode", "primary"]],
        on="particle_id",
    )

    # Calculate the transverse momentum (pt), polar angle (ptheta), pseudo-rapidity (peta),
    # and azimuthal angle (pphi) with the truth momentum information
    pt = np.sqrt(truth.tpx**2 + truth.tpy**2)
    ptheta = np.arctan2(pt, truth.tpz)
    peta = -np.log(np.tan(0.5 * ptheta))
    pphi = np.arctan2(truth.tpy, truth.tpx)

    # Add pt, ptheta, peta, pphi to the truth DataFrame
    truth = truth.assign(pt=pt, ptheta=ptheta, peta=peta, pphi=pphi)

    # Merge the relevant columns from the tubes DataFrame into the hits DataFrame
    hits = hits.merge(
        tubes[["hit_id", "isochrone", "skewed", "sector_id"]], on="hit_id"
    )

    # Calculate the transverse distance (r), azimuthal angle (phi), polar angle (theta), and pseudo-rapidity (eta)
    r = np.sqrt(hits.x**2 + hits.y**2)  # Transverse distance from the interaction point
    phi = np.arctan2(hits.y, hits.x)  # Azimuthal angle
    r3 = np.sqrt(
        hits.x**2 + hits.y**2 + hits.z**2
    )  # 3D distance from the interaction point
    theta = np.arccos(hits.z / r3)  # Polar angle
    eta = -np.log(np.tan(theta / 2.0))  # Pseudo-rapidity

    # Add r, phi, theta, eta to the hits DataFrame
    hits = hits.assign(r=r, phi=phi, theta=theta, eta=eta)

    # Merge the relevant columns from the truth DataFrame into the hits DataFrame
    hits = hits.merge(truth, on="hit_id")

    # skip noise hits.
    if not noise:
        truth = truth.query("primary==1")

    # skip skewed tubes
    if not skewed:
        hits = hits.query("skewed==0")

    # Restore the original order
    hits = hits.sort_values(by="original_order").reset_index(drop=True)

    # Drop the original_order column as it is no longer needed
    hits = hits.drop(columns=["original_order"])

    # Add 'event_id' column to this event.
    hits = hits.assign(event_id=event_id)

    return hits


def build_event(
    event_id: int,
    file_reader: ROOTFileReader,
    true_edge_method: str,
    input_edge_method: str,
    **kwargs,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Function that uses the provided hit information to build the true and input edges for a single event.

    Args:
        event_id (int): ID of the event to be processed.
        file_reader (ROOTFileReader): ROOTFileReader object containing the open ROOT file.
        true_edge_method (str): Method how to build the true edges. Options: 'layerwise', 'modulewise', 'orderwise', 'timeOrdered'.
        input_edge_method (str): Method how to build the input edges. Options: 'oldLayerwise', 'newLayerwise', 'all'.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        Returns a tuple with the following 12 elements:

        - Hit features (r, phi, isochrone) scaled by feature_scale for each hit.
        - Particle ID for each hit.
        - Layer ID for each hit.
        - True edges between hits.
        - Input edges between hits.
        - Hit ID for each hit.
        - True transverse momentum for each hit.
        - True vertex position (vx, vy, vz) for each hit.
        - MC PDG code for each hit.
        - True polar angle for each hit.
        - True pseudo-rapidity for each hit.
        - True azimuthal angle for each hit.
        - Primary particle flag for each hit (1 = primary, 0 = secondary).
    """

    hits = select_hits(
        event_id, file_reader, noise=kwargs["noise"], skewed=kwargs["skewed"]
    )

    # Get list of all layers
    layers = hits.layer_id.to_numpy()

    # Select the method how to build the true edges
    if (
        true_edge_method == "layerwise"
    ):  # Get the true edges by connecting hits layer by layer while the
        # radial distance to the interaction point must increase
        true_edges, hits = get_layerwise_edges(hits)
        logging.info(
            f"Layerwise truth graph built for {event_id} with size {true_edges.shape}"
        )
    elif (
        true_edge_method == "modulewise"
    ):  # Get the true edges by connecting hits, where each
        # hit must increase in radial distance to the interaction point
        true_edges = get_modulewise_edges(hits)
        logging.info(
            f"Modulewise truth graph built for {event_id} with size {true_edges.shape}"
        )
    elif (
        true_edge_method == "orderwise"
    ):  # Get the true edges using the order of the hits
        true_edges = get_orderwise_edges(hits)
        logging.info(
            f"Orderwise truth graph built for {event_id} with size {true_edges.shape}"
        )
    elif (
        true_edge_method == "timeordered"
    ):  # Get the true edges using the true time order of the hits
        true_edges = get_time_ordered_true_edges(hits)
        logging.info(
            f"Time ordered truth graph built for {event_id} with size {true_edges.shape}"
        )
    else:
        logging.error(f"{true_edge_method} is not a valid method to build true graphs")
        exit(1)

    # Build input edges by connecting hits layer by layer. The distance of the hits
    # from the interaction point need to be radially increasing
    if input_edge_method == "oldLayerwise":
        input_edges = get_layerwise_graph(
            hits, filtering=kwargs["filtering"], inneredges=False
        )
        logging.info(
            f"Layerwise input graph built for {event_id} with size {input_edges.shape}"
        )
    # Same as oldLayerwise, but also builds edges between adjacent hits in the same layer
    elif input_edge_method == "newLayerwise":
        input_edges = get_layerwise_graph(
            hits, filtering=kwargs["filtering"], inneredges=True
        )
        logging.info(
            f"Layerwise input graph built for {event_id} with size {input_edges.shape}"
        )
    # Build input edges by connecting all hits to all other hits.
    elif input_edge_method == "all":
        input_edges = get_all_edges(hits)
        logging.info(
            f"All input graph built for {event_id} with size {input_edges.shape}"
        )
    else:
        logging.error(
            f"{input_edge_method} is not a valid method to build input graphs"
        )
        exit(1)

    # feature scale for X=[r,phi,z]
    feature_scale = [100, np.pi, 100]

    return (
        hits[["r", "phi", "isochrone"]].to_numpy() / feature_scale,
        hits.particle_id.to_numpy(),
        layers,
        true_edges,
        input_edges,
        hits["hit_id"].to_numpy(),
        hits.pt.to_numpy(),
        hits[["vx", "vy", "vz"]].to_numpy(),
        hits.pdgcode.to_numpy(),
        hits.ptheta.to_numpy(),
        hits.peta.to_numpy(),
        hits.pphi.to_numpy(),
        hits.primary.to_numpy(),
    )


def prepare_event(
    event_id: int,
    file_reader: ROOTFileReader,
    output_dir: str,
    true_edge_method: str,
    input_edge_method: str,
    overwrite: bool,
    **kwargs,
) -> None:
    """
    Main function that reads the hit data from a ROOT file, processes the data, creates input and
    true edges between the hits, and finally saves the features into a PyTorch file.

    Args:
        event_id (int): ID of the event to be processed.
        file_reader (ROOTFileReader): ROOTFileReader object containing the open ROOT file.
        output_dir (str): Directory where the PyTorch files will be saved.
        true_edge_method (str): Method how to build the true edges. Options: 'layerwise', 'modulewise', 'orderwise', 'timeOrdered'.
        input_edge_method (str): Method how to build the input edges. Options: 'oldLayerwise', 'newLayerwise', 'all'.
        overwrite (bool): If True, existing files in the output directory will be overwritten.
    """

    logging.info(f"Preparing event number {event_id}")

    # Prepare the output filename and check if it already exists
    output_filename = f"{output_dir}/event_{event_id}.pt"
    if not os.path.exists(output_filename) or overwrite:
        logging.info(f"Writing into {output_filename}")
    else:
        logging.warning(
            f"File {output_filename} already exists! Skipping event {event_id}..."
        )
        return

    # Build the event
    (
        X,
        pid,
        layers,
        true_edges,
        input_edges,
        hid,
        pt,
        vertex,
        pdgcode,
        ptheta,
        peta,
        pphi,
        primary,
    ) = build_event(
        event_id=event_id,
        file_reader=file_reader,
        true_edge_method=true_edge_method,
        input_edge_method=input_edge_method,
        **kwargs,
    )

    # Build the PyTorch Geometric (PyG) 'Data' object
    data = Data(
        x=torch.from_numpy(X).float(),
        pid=torch.from_numpy(pid),
        layers=torch.from_numpy(layers),
        hid=torch.from_numpy(hid),
        pt=torch.from_numpy(pt),
        vertex=torch.from_numpy(vertex),
        pdgcode=torch.from_numpy(pdgcode),
        ptheta=torch.from_numpy(ptheta),
        peta=torch.from_numpy(peta),
        pphi=torch.from_numpy(pphi),
        true_edges=torch.from_numpy(true_edges),
        primary=torch.from_numpy(primary),
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
