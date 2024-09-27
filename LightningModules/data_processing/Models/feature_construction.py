import os
import logging
import numpy as np
import uproot as up
from time import time

from functools import partial
from tqdm.contrib.concurrent import process_map

from ..feature_store_base import FeatureStoreBase
from ..utils.trackml_event_utils import prepare_event as trackml_prepare_event
from ..utils.panda_event_utils import prepare_event as panda_prepare_event
from ..utils.root_file_reader import ROOTFileReader
from ..utils.pandaRoot_event_utils import prepare_sim_event, prepare_digi_event


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

        # Get a list with all the input sim ROOT files
        sim_dir = self.input_dir + "/sim"
        sim_file_list = [
            sim_dir + "/" + file + ":pndsim"
            for file in os.listdir(sim_dir)
            if os.path.isfile(os.path.join(sim_dir, file))
        ]

        logging.info(f"Sim files: {sim_file_list}")

        # Get a list with all the input digi ROOT files
        digi_dir = self.input_dir + "/digi"
        digi_file_list = [
            digi_dir + "/" + file + ":pndsim"
            for file in os.listdir(digi_dir)
            if os.path.isfile(os.path.join(digi_dir, file))
        ]

        logging.info(f"Digi files: {digi_file_list}")

        sim_branches = [
            "STTPoint.fX",
            "STTPoint.fY",
            "STTPoint.fZ",
            "STTPoint.fTime",
            "STTPoint.fPx",
            "STTPoint.fPy",
            "STTPoint.fPz",
            "STTPoint.fTrackID",
            "MCTrack.fStartX",
            "MCTrack.fStartY",
            "MCTrack.fStartZ",
            "MCTrack.fPx",
            "MCTrack.fPy",
            "MCTrack.fPz",
            "MCTrack.fPoints",
            "MCTrack.fPdgCode",
            "MCTrack.fStartT",
            "MCTrack.fGeneratorFlags",
        ]

        digi_branches = [
            "STTHit.fRefIndex",
            "STTHit.fX",
            "STTHit.fY",
            "STTHit.fZ",
            "STTHit.fDetectorID",
            "STTHit.fTubeID",
            "STTHit.fDepCharge",
            "STTHit.fIsochrone",
        ]

        logging.info("Preparing sim events")
        for batch in up.iterate(
            sim_file_list,
            sim_branches,
            step_size="100 MB",
            num_workers=self.n_workers,
        ):
            prepare_sim_event()

        logging.info("Preparing digi events")
        for batch in up.iterate(
            digi_file_list,
            digi_branches,
            step_size="100 MB",
            num_workers=self.n_workers,
        ):
            prepare_digi_event()

        end_time = time()
        print(
            f"Feature construction complete. Time taken: {end_time - start_time:f} seconds."
        )
