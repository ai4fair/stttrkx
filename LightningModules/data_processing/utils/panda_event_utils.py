import os
import logging
import torch
import trackml.dataset
import numpy as np
import pandas as pd

from typing import Tuple

from torch_geometric.data import Data
from .heuristic_utils import get_layerwise_graph, get_all_edges, graph_intersection
from ..utils.event_utils import (
    get_layerwise_edges,
    get_modulewise_edges,
    get_orderwise_edges,
    get_time_ordered_true_edges,
    process_particles,
)
from ..utils.ROOTFileReader import ROOTFileReader


def select_hits(
    event_number: int, file_reader: ROOTFileReader, noise: bool, skewed: bool, **kwargs
) -> pd.DataFrame:

    hits, tubes, particles, truth = file_reader.load_event(event_number)

    # store original order (needed for orderwise_true_edges function)
    hits["original_order"] = hits.index

    # preprocess 'particles' to get nhits, and drop duplicates
    particles["nhits"] = particles.groupby(["particle_id"])["nhits"].transform("count")
    particles.drop_duplicates(inplace=True, ignore_index=True)

    # apply particle selection (not nice, but works)
    if kwargs["selection"]:
        particles = process_particles(particles)

    # skip noise hits.
    if noise:
        # runs if noise=True
        truth = truth.merge(
            particles[["particle_id", "vx", "vy", "vz", "pdgcode"]],
            on="particle_id",
            how="left",
        )
    else:
        # runs if noise=False
        truth = truth.merge(
            particles[["particle_id", "vx", "vy", "vz", "pdgcode"]],
            on="particle_id",
            how="inner",
        )

    # calculate pt, ptheta, peta, pphi
    pt = np.sqrt(truth.tpx**2 + truth.tpy**2)
    ptheta = np.arctan2(pt, truth.tpz)
    peta = -np.log(np.tan(0.5 * ptheta))
    pphi = np.arctan2(truth.tpy, truth.tpx)

    # assign pt, ptheta, peta, pphi to truth
    truth = truth.assign(pt=pt, ptheta=ptheta, peta=peta, pphi=pphi)

    # merge some columns of tubes to the hits
    hits = hits.merge(
        tubes[["hit_id", "isochrone", "skewed", "sector_id"]], on="hit_id"
    )

    # skip skewed tubes
    if skewed is False:
        hits = hits.query("skewed==0")

    # Calculate derived variables from 'hits'
    r = np.sqrt(hits.x**2 + hits.y**2)  # Transverse distance from the interaction point
    phi = np.arctan2(hits.y, hits.x)  # Azimuthal angle
    r3 = np.sqrt(
        hits.x**2 + hits.y**2 + hits.z**2
    )  # 3D distance from the interaction point
    theta = np.arccos(hits.z / r3)  # Polar angle
    eta = -np.log(np.tan(theta / 2.0))  # Pseudo-rapidity

    # Add r, phi, theta, eta to 'hits' and merge with 'truth'
    hits = hits.assign(r=r, phi=phi, theta=theta, eta=eta).merge(truth, on="hit_id")

    # Restore the original order
    hits = hits.sort_values(by="original_order").reset_index(drop=True)

    # Drop the original_order column as it is no longer needed
    hits = hits.drop(columns=["original_order"])

    # Add 'event_id' column to this event.
    hits = hits.assign(event_id=event_number)

    return hits


def build_event(
    event_number: int,
    file_reader: ROOTFileReader,
    true_edge_method: str,
    input_edge_method: str,
    noise: bool,
    skewed: bool,
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
]:

    # Load event using "event_file" prefix (load_event function transferred to select_hits function).
    # hits, tubes, particles, truth = trackml.dataset.load_event(event_file)

    # Select hits, add new/select columns, add event_id
    hits = select_hits(event_number, file_reader, noise=noise, skewed=skewed, **kwargs)

    # Get list of all layers
    layers = hits.layer_id.to_numpy()

    if (
        true_edge_method == "layerwise"
    ):  # Get true edge list using the ordering of layers
        true_edges, hits = get_layerwise_edges(hits)
        logging.info(
            f"Layerwise truth graph built for {event_number} with size {true_edges.shape}"
        )
    elif true_edge_method == "modulewise":  # Get true edge list without layer ordering
        true_edges = get_modulewise_edges(hits)
        logging.info(
            f"Modulewise truth graph built for {event_number} with size {true_edges.shape}"
        )
    elif (
        true_edge_method == "orderwise"
    ):  # Get true edge list without layer ordering (natural order)
        true_edges = get_orderwise_edges(hits)
        logging.info(
            f"Orderwise truth graph built for {event_number} with size {true_edges.shape}"
        )
    elif (
        true_edge_method == "timeOrdered"
    ):  # Get true edge list without layer ordering (time ordered)
        true_edges = get_time_ordered_true_edges(hits)
        logging.info(
            f"Time ordered truth graph built for {event_number} with size {true_edges.shape}"
        )
    else:
        logging.error(f"{true_edge_method} is not a valid method to build true graphs")
        exit(1)

    # Get input edge list using order of layers.
    if input_edge_method == "oldLayerwise":
        input_edges = get_layerwise_graph(
            hits, filtering=kwargs["filtering"], inneredges=False
        )  # w/o samelayer edges
        logging.info(
            f"Layerwise input graph built for {event_number} with size {input_edges.shape}"
        )
    elif input_edge_method == "newLayerwise":
        input_edges = get_layerwise_graph(
            hits, filtering=kwargs["filtering"], inneredges=True
        )  # with samelayer edges
        logging.info(
            f"Layerwise input graph built for {event_number} with size {input_edges.shape}"
        )
    elif input_edge_method == "all":
        input_edges = get_all_edges(hits)
        logging.info(
            f"All input graph built for {event_number} with size {input_edges.shape}"
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
    )


def prepare_event(
    event_number: int,
    file_reader: ROOTFileReader,
    output_dir: str,
    true_edge_method: str,
    input_edge_method: str,
    noise: bool,
    skewed: bool,
    overwrite: bool,
    **kwargs,
) -> None:

    logging.info(f"Preparing event number {event_number}")

    output_filename = f"{output_dir}/event_{event_number}.pt"

    if not os.path.exists(output_filename) or overwrite:
        logging.info(f"Writing into {output_filename}")
    else:
        logging.warning(
            f"File {output_filename} already exists! Skipping event {event_number}..."
        )
        return

    # build event
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
    ) = build_event(
        event_number=event_number,
        file_reader=file_reader,
        true_edge_method=true_edge_method,
        input_edge_method=input_edge_method,
        noise=noise,
        skewed=skewed,
        **kwargs,
    )

    # build PyTorch Geometric (PyG) 'Data' object
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
        event_file=event_number,
    )

    # get input graph
    input_edges = torch.from_numpy(input_edges)
    true_edges = data.true_edges
    new_input_edges, y = graph_intersection(input_edges, true_edges)
    data.edge_index = new_input_edges
    data.y_pid = y  # if regime: [[pid]] points to filtering

    with open(output_filename, "wb") as output_file:
        torch.save(data, output_file)
