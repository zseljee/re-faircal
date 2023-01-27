from sklearn.cluster import KMeans
import numpy as np

from argparse import Namespace
from dataset import Dataset

from calibrationMethods import BetaCalibration

def oracle(dataset: Dataset, conf: Namespace) -> np.ndarray:
    """
    Run the Oracle algorithm. It sets up a BetaCalibration instance on each
    subgroup and then uses that to predict a calibrated score for that subgroup

    If images in the pair come from different subgroups (ie cross-ethnicity)
    the predicted score will be zero. Only same-ethnicity pairs will be considered.

    Parameters:
        dataset: Dataset - A dataset instance
        conf: Namespace - Configuration of current approach

    Returns:
        score: np.ndarray - Score of used approach
    """
    # Copy dataframe and set up test and score columns
    df = dataset.df.copy()
    df['test'] = df['fold'] == dataset.fold
    df['score'] = df[dataset.feature]

    # Compute calibrated score here, start as zeros
    calibrated_score = np.zeros_like(df['score'], dtype='float')

    # Get each combination of sensitive attributes (in this case only consider ethnicity)
    print("Computing results for subgroups...")
    for subgroup in dataset.iterate_subgroups():

        # Set up select mask for left and right image, initialize as all True
        select = np.full_like(calibrated_score, True, dtype=bool) # TODO this can be cleaner

        # Iterate attributes of subgroup
        for attribute in subgroup:

            # Get columns of current attribute, is always only 2 (for left and right image)
            for col in dataset.consts['sensitive_attributes'][attribute]['cols']:
                # Update masks for both images of current attribute value
                select &= (df[col] == subgroup[attribute])

        # Mask to select the train data of the above select
        select_train = select & (df['test'] == False)

        # Set up calibrator on train set of current subgroup
        calibrator = BetaCalibration(df['score'][select_train], df['same'][select_train], score_min=-1, score_max=1)

        # Use calibrator on all data
        calibrated_score[select] = calibrator.predict(df['score'][select])

    return calibrated_score
