import numpy as np

from dataset import Dataset
from argparse import Namespace


def uncalibrated(dataset: Dataset, conf: Namespace) -> np.ndarray:
    """
    Dummy function to simulate an approach, while not doing anything.
    Simply returns the 'score' column of the dataset.

    Parameters:
        dataset: Dataset - A dataset instance
        conf: Namespace - Configuration of current approach

    Returns:
        score: np.ndarray - Score of used approach
    """
    print("Running uncalibrated dummy approach...")

    return dataset.df[dataset.feature].to_numpy()
