import uproot as up
import pandas as pd
from typing import Tuple, List


class ROOTFileReader:
    """
    Class to read data from ROOT files using uproot and convert it into pandas DataFrames.
    """

    def __init__(self, root_file_path: str):
        """
        Initializes the ROOTFileReader by using the given path to the
        root file to open it using uproot.

        Args:
            root_file_path (str): Path to the ROOT file to read.
        """

        # Open the ROOT file using uproot
        try:
            with up.open(root_file_path) as f:
                self.root_file = f
        except FileNotFoundError:
            raise Exception(f"The file '{root_file_path}' does not exist.")

    def load_event(
        self,
        event_num: int,
        tree_names: List[str] = ["hits", "cells", "particles", "truth"],
    ) -> Tuple[pd.DataFrame, ...]:
        """
        Converts the TTrees in a ROOT file into pandas DataFrames for a single event.

        Either reads an event from the default hit (hits, cells) and truth (particles, truth) tree names or user-defined tree names.
        If read_truth is False, only the hit trees are read. For every tree a pandas DataFrame is created using the save_tree_entry_as_pandas function.

        Args:
            event_num (int): Number of the event to read. Corresponds to the entry number in the ROOT file.
            hit_tree_names (List[str], optional): List with the names of the TTrees to be read. Defaults to ["hits", "cells", "particles", "truth"].

        Returns:
            data_frames (Tuple[pd.DataFrame, ...]): Tuple of pandas DataFrames, one for each tree read.
        """

        # Create a pandas DataFrame for each tree and return them as a tuple
        return tuple(
            self._save_tree_entry_as_pandas(tree_name=tree_name, event_num=event_num)
            for tree_name in tree_names
        )

    def _save_tree_entry_as_pandas(
        self, tree_name: str, event_num: int
    ) -> pd.DataFrame:
        """
        Extracts all data from all TBranches in a TTree for a single event and saves it in a pandas DataFrame.

        This function expects that the TBranches contain a vector for each event, while the vector of each
        branch / column has the same length.

        Args:
            tree_name (str): Name of the TTree to read.
            event_num (int): Number of the event to read. Corresponds to the entry number in the ROOT file.

        Returns:
            pd.DataFrame: Pandas DataFrame containing the data from the TTree for the specified event.
        """

        # Get the names of all branches in the TTree
        branches = self.root_file[tree_name].keys()

        # Prepare the dictionary to store the branch data in
        tree_data = {}

        # Read the data from each branch and store it in the dictionary
        for branch in branches:
            branchData = self.root_file[tree_name + "/" + branch].array(
                entry_start=event_num, entry_stop=event_num + 1
            )
            # Save the contents of the branch in the dictionary
            tree_data[branch] = branchData[0]

        # Return the dictionary as a pandas DataFrame
        return pd.DataFrame(tree_data)

    def get_tree_entries(self, tree_name: str = "None") -> int:
        """
        Get the number of entries in a TTree in the ROOT file.
        If the tree_name is not specified, the number of entries of the
        first tree in the file is returned.

        Args:
            tree_name (str, optional): Name of the TTree. Defaults to "None".

        Returns:
            entries (int): Number of entries in the TTree.
        """

        if tree_name == "None":
            tree_names = self.root_file.keys()

        return self.root_file[tree_names[0]].num_entries
