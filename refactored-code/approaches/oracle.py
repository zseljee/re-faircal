from sklearn.cluster import KMeans
import numpy as np

from argparse import Namespace
from dataset import Dataset

from approaches.utils import get_threshold
from calibrationMethods import BetaCalibration

def oracle(dataset: Dataset, conf: Namespace) -> dict:
    data = {'df': dict(),
            'threshold': dict(),
            'fpr': dict()
           }

    dataset.select(None)
    
    df = dataset.df.copy()
    select_test = df['fold'] == dataset.fold
    df['test'] = select_test
    df['score'] = df[dataset.feature]

    calibrated_score = np.zeros_like(df['score'], dtype='float')

    print("Computing results for subgroups...")
    for subgroup in dataset.iterate_subgroups(use_attributes='ethnicity'):

        # TODO this can be cleaner?
        select = np.full_like(calibrated_score, True, dtype=bool)
        for col in dataset.consts['sensitive_attributes']['ethnicity']['cols']:
            select &= (df[col] == subgroup['ethnicity'])

        # Set up calibrator on train set of current subgroup
        calibrator = BetaCalibration(df['score'][select & ~select_test], df['same'][select & ~select_test], score_min=-1, score_max=1)

        # Use calibrator on all data
        calibrated_score[select] = calibrator.predict(df['score'][select])

        # Get threshhold using train set
        thr = get_threshold(calibrated_score[select & ~select_test], df['same'][select & ~select_test], conf.fpr_thr)
        fpr = 0. # TODO get FPR using test set

        data['threshold'][subgroup['ethnicity']] = thr
        data['fpr'][subgroup['ethnicity']] = fpr

    df['calibrated_score'] = calibrated_score.copy()

    # Use calibrated test scores to set a threshold
    thr = get_threshold(df['calibrated_score'][~select_test], df['same'][~select_test], conf.fpr_thr)

    # Save calibrated scores on test set too
    fpr = 0. # TODO compute FPR for test set

    data['threshold'][f'global'] = thr
    data['fpr'][f'global'] = fpr

    # Save results
    keepCols = ['test', 'score', 'calibrated_score', 'ethnicity', 'pair', 'same']
    data['df'] = df[keepCols]

    return data