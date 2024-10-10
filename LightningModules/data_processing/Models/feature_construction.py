import os
import yaml
import logging
import numpy as np
import uproot as up
import pandas as pd
import awkward as ak
import fnmatch

from time import time
from functools import partial
from tqdm.contrib.concurrent import process_map

from ..feature_store_base import FeatureStoreBase
from ..utils.trackml_event_utils import prepare_event as trackml_prepare_event
from ..utils.panda_event_utils import prepare_event as panda_prepare_event
from ..utils.root_file_reader import ROOTFileReader
from ..utils.pandaRoot_event_utils import prepare_event as pandaRoot_prepare_event

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


class TrackMLFeatureStore(FeatureStoreBase):
    """
    Processing model to convert STT data into files ready for GNN training

    Description:
        This class is used to read data from csv files containing PANDA STT hit and MC truth information.
        This information is then used to create true and input graphs and saved in PyTorch geometry files.
        It is a subclass of FeatureStoreBase which in turn is a subclass of the PyTorch Lighting's LightningDataModule.
    """

    def __init__(self, hparams: dict):
        """
        Initializes the TrackMLFeatureStore class.

        Args:
            hparams (dict): Dictionary containing the hyperparameters for the feature construction / data processing
        """

        # Call the base class (FeatureStoreBase in feature_store_base.py) constructor with the hyperparameters as arguments
        super().__init__(hparams)

        # self.detector_path = self.hparams["detector_path"]

    def prepare_data(self):
        """
        Main function for the feature construction / data processing.

        Description:
            Parallelizes the processing of input files by splitting them into
            evenly sized chunks and processing each chunk in parallel.
        """

        start_time = time()

        # Create the output directory if it does not exist
        logging.info("Writing outputs to " + self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        logging.info("Using the TrackMLFeatureStore to process data from CSV files.")

        # Find the input files
        all_files = os.listdir(self.input_dir)
        all_events = sorted(
            np.unique([os.path.join(self.input_dir, event[:15]) for event in all_files])
        )[: self.n_files]

        # Split the input files by number of tasks and select my chunk only
        all_events = np.array_split(all_events, self.n_tasks)[self.task]

        # Process input files with a worker pool and progress bar
        # Use process_map() from tqdm instead of mp.Pool from multiprocessing.
        process_func = partial(trackml_prepare_event, **self.hparams)
        process_map(
            process_func,
            all_events,
            max_workers=self.n_workers,
            chunksize=self.chunksize,
        )

        # Print the time taken for feature construction
        end_time = time()
        print(
            f"Feature construction complete. Time taken: {end_time - start_time:f} seconds."
        )


class PandaFeatureStore(FeatureStoreBase):
    """
    Class to process ROOT files containing PANDA STT data and save the processed tensors into PyTorch files.
    """

    def __init__(self, hparams: dict) -> None:
        """
        Default constructor for the PandaFeatureStore class.

        Initializes the PandaFeatureStore class by calling the FeatureStoreBase constructor with a dictionary containing the hyperparameters.

        Args:
            hparams (dict): Dictionary containing the hyperparameters for the PANDA data processing.
        """

        # Call the base class (FeatureStoreBase in feature_store_base.py) constructor with the hyperparameters as arguments
        super().__init__(hparams)

    def prepare_data(self) -> None:
        """
        Main method for the PANDA data processing.
        """

        # Start the timer to measure the time taken for feature construction
        start_time = time()

        # Create the output directory if it does not exist yet
        logging.info("Writing outputs to " + self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        # Check if the input file is a ROOT file by examining the extension
        fileExtension = os.path.splitext(self.input_dir)[1]
        if fileExtension != ".root":
            logging.error(f"Specified input file {self.input_dir} is not a ROOT file!")
            raise Exception("Input file must be a ROOT file.")

        # Open the input ROOT file using the ROOTFileReader class and get the number of events saved in the file
        root_file_reader = ROOTFileReader(self.input_dir)
        total_events = root_file_reader.get_tree_entries()
        logging.info(f"Total number of events in the file: {total_events}")

        # Get the number of events to process
        # If this hyperparameter is not specified, process all events in the file
        if "n_files" not in self.hparams.keys():
            nEvents = total_events
        else:
            nEvents = self.hparams["n_files"]

        logging.info(f"Number of events to process: {nEvents}")

        # make iterable of events
        all_events = range(nEvents)

        # Define a new function by passing the static arguments to the prepare_event function
        process_func = partial(
            panda_prepare_event, file_reader=root_file_reader, **self.hparams
        )

        # Execute the new process_func in parallel for each event withing the all_events iterable
        process_map(
            process_func,
            all_events,
            max_workers=self.n_workers,
            chunksize=self.chunksize,
        )

        # Print the time taken for feature construction
        end_time = time()
        print(
            f"Feature construction complete. Time taken: {end_time - start_time:f} seconds."
        )


class PandaRootFeatureStore(FeatureStoreBase):
    """
    Class to process data from PandaRoot sim and digi files and save the processed tensors into PyTorch files.
    """

    def __init__(self, hparams: dict) -> None:
        """
        Default constructor for the PandaRootFeatureStore class.

        Initializes the PandaRootFeatureStore class by calling the FeatureStoreBase constructor with a dictionary containing the hyperparameters.

        Args:
            hparams (dict): Dictionary containing the hyperparameters for the PandaRoot data processing.
        """

        # Call the base class (FeatureStoreBase in feature_store_base.py) constructor with the hyperparameters as arguments
        super().__init__(hparams)

    def prepare_data(self) -> None:
        """
        Main method for the PandaRoot data processing.
        """

        # Start the timer to measure the time taken for feature construction
        start_time = time()

        # Create the output directory if it does not exist yet
        logging.info("Writing outputs to " + self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        # Save the STT geometry data from the csv file into a pandas data frame
        stt_geo_df = pd.read_csv(
            "/home/nikin105/mlProject/data/detectorGeometries/tubePos.csv"
        )

        # Read the signal signature from the YAML file.
        with open(self.hparams["signal_signature_file"], "r") as file:
            signal_signature = yaml.safe_load(file)

        # Count the number of input sim and digi ROOT files
        num_sim_files = 0
        for filename in os.listdir(self.input_dir + "/sim"):
            if fnmatch.fnmatch(filename, "*_sim.root"):
                num_sim_files += 1
        logging.info(f"Number of sim files: {num_sim_files}")

        num_digi_files = 0
        for filename in os.listdir(self.input_dir + "/digi"):
            if fnmatch.fnmatch(filename, "*_digi.root"):
                num_digi_files += 1
        logging.info(f"Number of digi files: {num_digi_files}")

        # Check if the number of sim and digi files are the same
        if num_sim_files != num_digi_files:
            logging.error(
                f"Number of sim files ({num_sim_files}) and digi files ({num_digi_files}) do not match!"
            )
            raise Exception("Number of sim and digi files must match.")

        # TBranches of the simulation file and the corresponding names in the data frame
        sttPoint_branch_dict = {
            "STTPoint.fX": "tx",
            "STTPoint.fY": "ty",
            "STTPoint.fZ": "tz",
            "STTPoint.fTime": "tT",
            "STTPoint.fPx": "tpx",
            "STTPoint.fPy": "tpy",
            "STTPoint.fPz": "tpz",
            "STTPoint.fTrackID": "particle_id",
        }

        mcTrack_branch_dict = {
            "MCTrack.fStartX": "vx",
            "MCTrack.fStartY": "vy",
            "MCTrack.fStartZ": "vz",
            "MCTrack.fPdgCode": "pdgcode",
            "MCTrack.fProcess": "process_code",
            "MCTrack.fMotherID": "mother_id",
            "MCTrack.fSecondMotherID": "second_mother_id",
        }

        # TBranches of the digitalization file and the corresponding names in the data frame
        sttHit_branch_dict = {
            "STTHit.fRefIndex": "hit_id",
            "STTHit.fX": "x",
            "STTHit.fY": "y",
            "STTHit.fZ": "z",
            "STTHit.fDetectorID": "volume_id",
            "STTHit.fTubeID": "module_id",
            "STTHit.fIsochrone": "isochrone",
        }

        key_dict = {
            "sttPoint": [
                sttPoint_branch_dict[sttPoint_key]
                for sttPoint_key in sttPoint_branch_dict.keys()
            ],
            "mcTrack": [
                mcTrack_branch_dict[mcTrack_key]
                for mcTrack_key in mcTrack_branch_dict.keys()
            ],
            "sttHit": [
                sttHit_branch_dict[sttHit_key]
                for sttHit_key in sttHit_branch_dict.keys()
            ],
        }

        events_processed = 0

        # Iterate over all files
        for file_num in range(self.n_files):

            logging.info(f"Processing File {file_num+1} of {self.n_files}")

            # Open the simulation file using uproot
            sim_file_name = (
                self.input_dir
                + "/sim/"
                + self.hparams["prefix"]
                + f"_{file_num}_sim.root:pndsim"
            )
            logging.debug(f"Simulation file:\n{sim_file_name}")
            sim_file = up.open(sim_file_name, num_workers=self.n_workers)

            # Create an iterator that contains a chunk of the simulation events
            sim_iterator = sim_file.iterate(
                expressions=list(mcTrack_branch_dict.keys())
                + list(sttPoint_branch_dict.keys()),
                library="pd",
                step_size=10000,
            )
            logging.debug(f"Simulation events iterator:\n{sim_iterator}")

            digi_file_name = (
                self.input_dir
                + "/digi/"
                + self.hparams["prefix"]
                + f"_{file_num}_digi.root:pndsim"
            )
            logging.debug(f"Digitalization file:\n{digi_file_name}")
            digi_file = up.open(digi_file_name, num_workers=self.n_workers)

            # Create an iterator that contains a chunk of the digitalization events
            digi_iterator = digi_file.iterate(
                expressions=sttHit_branch_dict.keys(),
                library="pd",
                step_size=10000,
            )
            logging.debug(f"Digitalization iterator:\n{digi_iterator}")

            # Iterate over the simulation and digitalization chunks
            for chunk, digi_chunk in zip(sim_iterator, digi_iterator):

                chunk = chunk.rename(columns=mcTrack_branch_dict)
                chunk = chunk.rename(columns=sttPoint_branch_dict)
                digi_chunk = digi_chunk.rename(columns=sttHit_branch_dict)

                logging.debug(f"Simulation chunk:\n{chunk}")
                logging.debug(f"Digitalization chunk:\n{digi_chunk}")
                last_event_num = events_processed + len(chunk)
                event_ids = np.arange(events_processed, last_event_num, dtype=int)

                # Combine the simulation and digitalization chunks into a single chunk
                chunk = pd.concat([chunk, digi_chunk], axis=1)
                chunk = chunk.assign(event_id=event_ids)
                logging.debug(f"Chunk:\n{chunk}")
                del digi_chunk

                # create an iterator for the rows of the data frame
                row_iterator = tqdm(chunk.itertuples(index=False), total=len(chunk))

                process_func = partial(
                    pandaRoot_prepare_event,
                    key_dict=key_dict,
                    signal_signatures=signal_signature,
                    stt_geo=stt_geo_df,
                    **self.hparams,
                )

                process_map(process_func, row_iterator, max_workers=self.n_workers, chunksize=self.chunksize)

                with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                    executor.map(process_func, row_iterator)

                events_processed += len(chunk)

            sim_file.close()
            digi_file.close()

        end_time = time()

        print(
            f"Feature construction complete. Time taken: {end_time - start_time:f} seconds."
        )
