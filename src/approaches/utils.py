import numpy as np

from sklearn.metrics import roc_curve
from argparse import Namespace

from dataset import Dataset


def get_ks(confidences, ground_truth):
    """
    KS Divergence Score
    Proposed by
    Kartik Gupta, Amir Rahimi, Thalaiyasingam Ajanthan, Thomas Mensink, Cristian Sminchisescu,
    and Richard Hartley. Calibration of neural networks using splines. In International
    Conference on Learning Representations, 2021.
    URL https://openreview.net/forum?id=eQe8DEWNN2W.

    Copied from
    https://github.com/tiagosalvador/faircal/blob/a9a0f1852275b68b027edff2fd9992d32e829ad6/utils.py#L48
    """
    n = len(ground_truth)
    order_sort = np.argsort(confidences)
    ks = np.max(np.abs(np.cumsum(confidences[order_sort])/n-np.cumsum(ground_truth[order_sort])/n))
    return ks


def get_brier(confidences, ground_truth):
    """
    Brier Score
    Proposed by
    Morris H DeGroot and Stephen E Fienberg. The comparison and evaluation of forecasters.
    Journal of the Royal Statistical Society: Series D (The Statistician), 32(1-2):12-22, 1983.

    Copied from
    https://github.com/tiagosalvador/faircal/blob/a9a0f1852275b68b027edff2fd9992d32e829ad6/utils.py#L55
    """
    # Compute Brier Score
    brier = np.zeros(confidences.shape)
    brier[ground_truth] = (1-confidences[ground_truth])**2
    brier[np.logical_not(ground_truth)] = (confidences[np.logical_not(ground_truth)])**2
    brier = np.mean(brier)
    return brier


def get_metrics(confidences: np.ndarray, dataset: Dataset, conf: Namespace) -> dict:
    """
    Given a list of calibrated scores, compute some metrics indicating its
    accuracy and fairness.

    Parameters:
        confidences: np.ndarray - calibrated scores of size N,
        dataset: Dataset - Complete dataset of size N
        conf: Namespace - Configuration that determined how the calibrated scores came to be

    Returns:
        data: dict[str, dict[str, any]] - A dictionary with subgroups as keys and metrics as values
        Subgroups include 'Global' as overal metrics and one key per subgroup, made by concatenating
        subgroup values with intermediate '_' (ie 'Asian_Female'). Metrics include
        - fpr, tpr, thr: np.ndarray (each) - FPR, TPR and thresholds computed using sklearn.metrics.roc_curve
        - ks: float - KS score
        - brier: float - brier score
    """
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
        'brier': get_brier(confidences, ground_truth),
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
            'brier': get_brier(confidences[select], ground_truth[select]),
        }

    return data


def thr_at_fpr_from_score(score, ground_truth, target_fpr):
    # Get metrics of uncalibrated scores
    fpr, _, thr = roc_curve(y_true=ground_truth,
                            y_score=score,
                            drop_intermediate=False)

    # Find the threshold at which the FPR is the pre-defined target FPR
    return thr_at_fpr(thr, fpr, target_fpr)


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
