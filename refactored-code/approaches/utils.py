from sklearn.metrics import roc_curve
import numpy as np

def get_threshold(scores: np.ndarray, ground_truth: np.ndarray, target_fpr: float=0.05) -> float:
    """
    Given a list of scores and the ground truth, return the threshold at which point
    the FPR is closest to the target_fpr.

    First compute the ROC curve using sklearn.metrics.roc_curve, then choose the threshold
    where the FPR is closest to the target FPR.

    Parameters:
        scores: np.ndarray - A np array of shape (n,), given as y_score in roc_curve
        ground_truth: np.ndarray - A np array of shape (n,) with target classes
    
    Returns:
        thr: float - Threshold at which the FPR for the given data is closest to the target FPR
    """

    # Compute FPRs and thresholds
    fpr, _, thr = roc_curve(y_score=scores,
                            y_true=ground_truth,
                            drop_intermediate=False,
                            pos_label=True
                           )
    # Get index of item that is closest to the target FPR
    thr_idx = np.argmin(np.abs(fpr-target_fpr))

    # Return the corresponding threshold
    return thr[thr_idx]