import numpy as np

from dataset import Dataset
from argparse import Namespace

from calibrationMethods import BetaCalibration
from approaches.utils import get_threshold

def baseline(dataset: Dataset, conf: Namespace):
    data = {'confidences': dict(),
            'threshold': dict(),
            'fpr': dict()
           }

    print("Calibrating global scores...")

    dataset.select(None)
    df = dataset.df.copy()
    select_test = df['fold'] == dataset.fold
    df['test'] = select_test
    df['score'] = df[dataset.feature]
    df['calibrated_score'] = 0.

    score_train = df['score'][~select_test]
    gt_train = df['same'][~select_test]

    calibrator = BetaCalibration(scores=score_train,
                                 ground_truth=gt_train,
                                 score_min=-1,
                                 score_max=1,
                                )
    df['calibrated_score'] = calibrator.predict(df['score'])
    calscore_train = df['calibrated_score'][~select_test]

    thr = get_threshold(calscore_train, gt_train, conf.fpr_thr)
    fpr = 0. # TODO compute FPR for test set

    data['threshold']['global'] = thr
    data['fpr']['global'] = fpr

    
    print("Calibrating subgroup scores...")
    for subgroup in dataset.iterate_subgroups(use_attributes='ethnicity'):

        # TODO this can be cleaner?
        select = np.copy(~select_test)
        for col in dataset.consts['sensitive_attributes']['ethnicity']['cols']:
            select &= (df[col] == subgroup['ethnicity'])
            
        thr = get_threshold(df['calibrated_score'][select], df['same'][select], conf.fpr_thr)
        fpr = 0. # TODO

        data['threshold'][subgroup['ethnicity']] = thr
        data['fpr'][subgroup['ethnicity']] = fpr
    
    # Save results
    keepCols = ['test', 'score', 'calibrated_score', 'ethnicity', 'pair', 'same']
    data['df'] = df[keepCols]

    return data