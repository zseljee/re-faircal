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

    df = dataset.df.copy()

    # Only take test data
    if dataset.fold is not None:
        select_test = df['fold'] == dataset.fold
        df = df[select_test].copy()
        confidences = confidences[select_test]

    ground_truth = df['same'].astype(int).to_numpy()

    # Global results
    fpr, tpr, thr = roc_curve(y_true=ground_truth,
                              y_score=confidences,
                              drop_intermediate=False)

    data['Global'] = {
        'fpr': fpr,
        'tpr': tpr,
        'thr': thr,
        'ks': get_ks(confidences, ground_truth),
        'brier': get_brier(confidences, ground_truth)
    }

    for subgroup in dataset.iterate_subgroups():

        select = np.full_like(confidences, True, dtype=bool)
        for attribute in subgroup:
            for col in dataset.consts['sensitive_attributes'][attribute]['cols']:
                select &= (df[col] == subgroup[attribute])
            
        subgroup_key = '_'.join(subgroup.values())

        fpr, tpr, thr = roc_curve(y_true=ground_truth[select],
                                  y_score=confidences[select],
                                  drop_intermediate=False)

        data[subgroup_key] = {
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
    return np.interp(target_fpr, fpr, thr)


def tpr_at_fpr(tpr, fpr, target_fpr):
    """
    Interpolate fpr->tpr to find fpr for a target fpr

    Parameters:
        thr: np.ndarray - A 1D np array containing thresholds
        fpr: np.ndarray - A 1D np array of the same size with corresponding FPRs
        target_fpr: float - A target FPR

    Returns:
        tpr: float - TPR at target FPR
    """

    # Return the corresponding threshold
    return np.interp(target_fpr, fpr, tpr)
