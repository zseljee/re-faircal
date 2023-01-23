from sklearn.metrics import roc_curve
from argparse import Namespace
import numpy as np
import pandas as pd

from dataset import Dataset

def get_ks(confidences, ground_truth):
    n = len(ground_truth)
    order_sort = np.argsort(confidences)
    ks = np.max(np.abs(np.cumsum(confidences[order_sort])/n-np.cumsum(ground_truth[order_sort])/n))
    return ks


def get_brier(confidences, ground_truth):
    # Compute Brier Score
    brier = np.zeros(confidences.shape)
    brier[ground_truth] = (1-confidences[ground_truth])**2
    brier[np.logical_not(ground_truth)] = (confidences[np.logical_not(ground_truth)])**2
    brier = np.mean(brier)
    return brier

def get_metrics(confidences: np.ndarray, dataset: Dataset, conf: Namespace) -> dict:
    data = dict()

    subgroups = ['Global',] + dataset.consts['sensitive_attributes']['ethnicity']['values']
    subgroupCols = dataset.consts['sensitive_attributes']['ethnicity']['cols']

    df = dataset.df.copy()
    df['test'] = (df['fold'] == dataset.fold)

    ground_truth = df['same'].astype(int).to_numpy()

    for subgroup in subgroups:

        select = (df['test'] == True)
        if subgroup != 'Global':
            for col in subgroupCols:
                select &= (df[col] == subgroup)

        fpr, tpr, thr = roc_curve(y_true=ground_truth[select],
                                  y_score=confidences[select],
                                  drop_intermediate=False)

        data[subgroup] = {
            'fpr': fpr,
            'tpr': tpr,
            'thr': thr,
            'ks': get_ks(confidences[select], ground_truth[select]),
            'brier': get_brier(confidences[select], ground_truth[select])
        }

    return data


def thr_at_fpr(thr, fpr, target_fpr):
    """
    Given a list of thresholds and FPR at those threshold, give the threshold
    that gives results closest to the target FPR

    Parameters:
        thr: np.ndarray - A 1D np array containing thresholds
        fpr: np.ndarray - A 1D np array of the same size with corresponding FPRs
        target_fpr: float - A target FPR

    Returns:
        thr: float - Threshold at which the FPR for the given data is closest to the target FPR
    """
    # Get index of item that is closest to the target FPR
    idx = np.argmin(np.abs(fpr-target_fpr))

    # Return the corresponding threshold
    return thr[idx]


def tpr_at_fpr(tpr, fpr, target_fpr):
    """
    Given a list of FPR and corresponding TPR, give the TPR where the corresponding
    FPR is closest to the target FPR.

    Parameters:
        thr: np.ndarray - A 1D np array containing thresholds
        fpr: np.ndarray - A 1D np array of the same size with corresponding FPRs
        target_fpr: float - A target FPR

    Returns:
        tpr: float - TPR at target FPR
    """
    # Get index of item that is closest to the target FPR
    idx = np.argmin(np.abs(fpr-target_fpr))

    # Return the corresponding threshold
    return tpr[idx]
