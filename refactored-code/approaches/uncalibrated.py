import numpy as np

from dataset import Dataset
from argparse import Namespace

from approaches.utils import get_threshold

def uncalibrated(dataset: Dataset, conf: Namespace):
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
    df['calibrated_score'] = df['score'].copy()

    thr = get_threshold(df['score'][~select_test], df['same'][~select_test], conf.fpr_thr)
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