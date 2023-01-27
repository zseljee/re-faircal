import numpy as np

from dataset import Dataset
from argparse import Namespace

from calibrationMethods import BetaCalibration

def baseline(dataset: Dataset, conf: Namespace) -> np.ndarray:
    """
    Run the baseline algorithm. It runs the BetaCalibration algorithm on
    the score of the current feature. It does not take into account
    any differences between subgroups.

    After calibrating, predict on all data, which gives an output
    of the same number of samples as the dataset

    Parameters:
        dataset: Dataset - A dataset instance
        conf: Namespace - Configuration of current approach

    Returns:
        score: np.ndarray - Score of used approach
    """

    print("Calibrating global scores...")
    # Copy dataset dataframe and set up the score and test columns
    df = dataset.df.copy()
    df['test'] = df['fold'] == dataset.fold
    df['score'] = df[dataset.feature]

    # Extract score and ground truth of train set
    score = df[ df['test'] == False ][ 'score' ].to_numpy()
    ground_truth = df[ df['test'] == False ][ 'same' ].to_numpy(dtype=int)

    # Set up calibrator on train set
    calibrator = BetaCalibration(scores=score,
                                 ground_truth=ground_truth,
                                 score_min=-1,
                                 score_max=1,
                                )

    # Run calibrator on all data
    return calibrator.predict(df['score'])
