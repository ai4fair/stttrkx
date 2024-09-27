import numpy as np
import uproot as up
import logging
import pdg


def get_branch_entry(entry_id: int, branch_name: str, tree: up.TTree) -> np.ndarray:
    """
    Get the values for one entry saved in a given TBranch of a given TTree as a numpy
    array.

    Args:
        entry_id (int): Entry number.
        branch_name (str): Name of the TBranch.
        tree (up.TTree): Uproot TTree object.

    Returns:
        np.ndarray: Array with the values of the given TBranch for the given entry.
    """

    return tree[branch_name].array(
        entry_start=entry_id, entry_stop=entry_id + 1, library="np"
    )[0]


def make_track_dict(
    track_ids: np.ndarray,
    track_parameter_arr: np.ndarray,
    track_dict: dict[str, np.ndarray] = {},
) -> dict[str, np.ndarray]:
    """
    Create a dictionary with track IDs as keys and track parameter arrays as values.

    Args:
        track_ids (np.ndarray): Array with track IDs for each MC point.
        track_parameter_arr (np.ndarray): Array with parameter values for each MC point.
        track_dict (dict[str, np.ndarray], optional): Dictionary with the same format
        as the output. Defaults to {}.

    Returns:
        dict[str, np.ndarray]: Dictionary with track IDs as keys and track parameter
                               arrays as values.
    """

    # Iterate over all MC points
    for mc_point_index in range(len(track_ids)):
        # Get the track ID of the current MC point
        track_id = track_ids[mc_point_index]
        # Check if the track ID is not in the dictionary, add it with the current track
        # parameter value
        if track_id not in track_dict:
            track_dict[track_id] = np.array([track_parameter_arr[mc_point_index]])
        # If the track ID is already in the dictionary, append the current track
        # parameter value
        else:
            track_dict[track_id] = np.append(
                track_dict[track_id], track_parameter_arr[mc_point_index]
            )

    # Return the updated track dictionary
    return track_dict


def get_particle_tex_name(
    pdg_id: int,
    pdg_db: str = "sqlite:////home/nikin105/mlProject/data/pdg/pdgall-2024-v0.1.0.sqlite",
) -> str:
    """
    Get the latex representation of a particle name based on its PDG ID. The PDG MC numbering
    scheme can be found here: https://pdg.lbl.gov/2024/reviews/rpp2024-rev-monte-carlo-numbering.pdf.
    "Nucleus" represents all nuclei. Nuclei have a 10-digit PDG ID and the numbering scheme is listed
    under point 16. of the document.

    Args:
        pdg_id (int): PDG ID of the particle.
        pdg_db (str, optional): Path to the PDG database. Defaults to "sqlite:////home/
        nikin105/mlProject/data/pdg/pdgall-2024-v0.1.0.sqlite".

    Returns:
        str: Latex representation of the particle name.
    """

    # Connect to the pdg database
    pdg_api = pdg.connect(pdg_db)

    # Dictionary with particle names and their corresponding latex representation
    particle_tex_names = {
        "p": r"p",
        "pbar": r"\bar{p}",
        "gamma": r"\gamma",
        "pi+": r"\pi^+",
        "pi-": r"\pi^-",
        "pi0": r"\pi^0",
        "e+": r"e^+",
        "e-": r"e^-",
        "Xibar+": r"\bar{\Xi}^+",
        "Xi-": r"\Xi^-",
        "Lambda0": r"\Lambda^0",
        "Lambdabar0": r"\bar{\Lambda}^0",
        "mu+": r"\mu^+",
        "mu-": r"\mu^-",
        "n": r"n",
        "K+": r"K^+",
        "K-": r"K^-",
        "Sigma+": r"\Sigma^+",
        "Sigma-": r"\Sigma^-",
        "Sigma0": r"\Sigma^0",
        "Sigmabar-": r"\bar{\Sigma}^-",
        "Sigmabar+": r"\bar{\Sigma}^+",
    }

    # Check if the particle ID is 8888 belonging to the ppbar system
    if pdg_id == 88888:
        return r"p\bar{p}"
    # Check if the particle ID is 10 digits long, which represents a nucleus
    elif len(str(pdg_id)) == 10:
        return "nucleus"
    # Else return the latex representation of the particle name
    elif pdg_api.get_particle_by_mcid(str(pdg_id)).name in particle_tex_names:
        return particle_tex_names[pdg_api.get_particle_by_mcid(str(pdg_id)).name]
    else:
        logging.warning(f"No latex conversion for PDG ID: {pdg_id}")
        return pdg_api.get_particle_by_mcid(str(pdg_id)).name


def get_all_mother_ids(
    mother_ids: np.ndarray[int], second_mother_ids: np.ndarray[int]
) -> np.ndarray[int]:
    """
    Uses the mother and second mother IDs proved by PandaRoot to create an array with all mother IDs.

    Args:
        mother_ids (np.ndarray): Array with the particle IDs of each mother particle.
        second_mother_ids (np.ndarray): Second array with the particle IDs of each mother particle.

    Returns:
        np.ndarray: Array with all mother IDs.
    """

    # Create an array combining the mother and second mother IDs
    all_mother_ids = np.zeros(len(mother_ids), dtype=int)

    # Iterate over all particles
    for particle_id in range(len(mother_ids)):
        # Get the mother ID and second mother ID of the current MC point
        mother_id = mother_ids[particle_id]
        second_mother_id = second_mother_ids[particle_id]
        # When the mother ID is -1, set the second mother ID as the mother ID
        if mother_id == -1:
            all_mother_ids[particle_id] = second_mother_id
        else:
            all_mother_ids[particle_id] = mother_id

    # Return the array with all mother IDs
    return all_mother_ids


def get_process_ids(
    process_ids: np.ndarray[int],
    mother_ids: np.ndarray[int],
    pdg_ids: np.ndarray[int],
    particle_id: int,
) -> tuple[np.ndarray[int], np.ndarray[int]]:
    """
    Get an array with the PDG Monte Carlo particle ID numbers and the VMC
    physics process code (https://root.cern/doc/v610/TMCProcess_8h_source.html)
    of the particle itself and of all mother particles of a given particle from
    first to last.

    Args:
        process_ids (np.ndarray[int]): Array with the VMC physics process codes of each particle.
        mother_ids (np.ndarray[int]): Array with the mother particle IDs of each particle.
        pdg_ids (np.ndarray[int]): Array with the PDG MC IDs of each particle.
        particle_id (int): ID of the daughter particle.

    Returns:
        tuple[np.ndarray[int], np.ndarray[int]]: Tuple with two arrays. The first one
        contains all PDG MC IDs of the process and the second one contains all VMC physics
        process codes.
    """

    # Create an array for the mc ids of the process leading to the current
    # particle
    process_mc_ids = np.array([], dtype=int)
    curr_process_ids = np.array([], dtype=int)

    # Iterate over the mother ids until the primary particle is reached (mother
    # id = -1)
    while particle_id != -1:
        # Insert the current particle's pdg id at the beginning of the process
        # mc id array
        process_mc_ids = np.insert(process_mc_ids, 0, [pdg_ids[particle_id]])
        curr_process_ids = np.insert(curr_process_ids, 0, [process_ids[particle_id]])
        # Set the current particle index to the mother id of the previous
        # particle
        particle_id = int(mother_ids[particle_id])

    # Return the array with all mc ids of the process
    return process_mc_ids, curr_process_ids


def get_process_tex_str(process_mc_ids: np.array, max_depth: int = np.inf) -> str:

    # Set an integer to count the current depth of the process
    current_depth = 1

    # Iterate over all particles in the process
    for particle_index in range(len(process_mc_ids)):
        # Get the latex representation of the particle name
        particle_tex_name = get_particle_tex_name(process_mc_ids[particle_index])
        # Set the name of the first particle with an $ to activate
        # math mode in latex instead of an arrow in front
        if particle_index == 0:
            process_tex_str = r"$" + particle_tex_name
        # Add an arrow in front of all other particles
        else:
            process_tex_str = process_tex_str + r" \to " + particle_tex_name
        # Increase the current depth by one
        current_depth += 1
        # Break the loop if the maximum depth or the last particle
        # is reached and append a closing $ to end math mode
        if current_depth > max_depth or particle_index == len(process_mc_ids) - 1:
            process_tex_str = process_tex_str + r"$"
            break

    # Return the latex formatted string of the process
    return process_tex_str


def is_signal_particle(
    process_mc_ids: np.ndarray[int],
    process_ids: np.ndarray[int],
    signal_mc_ids: list[list[int]],
    signal_process_ids: list[list[int]],
) -> bool:
    """
    Check if a given particle is a signal particle based on its PDG Monte Carlo IDs
    and the VMC physics process codes of the process.

    Args:
        process_mc_ids (np.ndarray[int]): Array with the PDG Monte Carlo IDs of the process.
        process_ids (np.ndarray[int]): Array with the VMC physics process codes of the process.
        signal_mc_ids (list[list[int]]): List with the PDG Monte Carlo IDs considered as signal.
        signal_process_ids (list[list[int]]): List with the VMC physics process codes considered as signal.

    Returns:
        bool: True if the particle is a signal particle, False otherwise.
    """

    # Check if the MC IDs are contained in the list for signal MC IDs
    has_signal_mc_ids = False

    for mc_ids in signal_mc_ids:
        if np.array_equal(mc_ids, process_mc_ids):
            has_signal_mc_ids = True
            break

    # Check if the process IDs are contained in the list for signal process IDs
    has_signal_process_ids = False

    for proc_id in signal_process_ids:
        if np.array_equal(proc_id, process_ids):
            has_signal_process_ids = True
            break

    # Return True if both the MC IDs and the process IDs are contained in the corresponding lists
    return has_signal_mc_ids and has_signal_process_ids


def get_process_name(process_id: int) -> str:
    """
    Get the name of a process based on its process ID.
    IDs taken from: https://root.cern/doc/v610/TMCProcess_8h_source.html

    Args:
        process_id (int): Process ID.

    Returns:
        str: String containing the name of the process.
    """

    process_names = [
        "Primary particle emission",
        "Multiple scattering",
        "Energy loss",
        "Bending in magnetic field",
        "Decay",
        "Lepton pair production",
        "Compton scattering",
        "Photoelectric effect",
        "Bremsstrahlung",
        "Delta ray",
        "Positron annihilation",
        "Positron annihilation at rest",
        "Positron annihilation in flight",
        "Hadronic interaction",
        "Nuclear evaporation",
        "Nuclear fission",
        "Nuclear absorption",
        "Antiproton annihilation",
        "Antineutron annihilation",
        "Neutron capture",
        "Hadronic elastic",
        "Hadronic incoherent elastic",
        "Hadronic coherent elastic",
        "Hadronic inelastic",
        "Photon inelastic",
        "Muon nuclear interaction",
        "Electron nuclear interaction",
        "Positron nuclear interaction",
        "Time of flight limit",
        "Nuclear photofission",
        "Rayleigh effect",
        "No active process",
        "Energy threshold",
        "Light absorption",
        "Light detection",
        "Light scattering",
        "Maximum allowed step",
        "Cerenkov production",
        "Cerenkov feed back photon",
        "Cerenkov photon reflection",
        "Cerenkov photon refraction",
        "Synchrotron radiation",
        "Scintillation",
        "Transportation",
        "Unknown process",
        "Coulomb scattering",
        "Photo nuclear interaction",
        "User defined process",
        "Optical photon wavelength shifting",
        "Transition radiation",
    ]

    return process_names[process_id]
