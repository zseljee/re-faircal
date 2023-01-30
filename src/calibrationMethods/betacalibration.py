"""This is a partial copy of the file of the code published by Salvador et.al.

The code below is exactly the same as the file `calibration_methods.py` except
that only the betacalibration method has been kept.
"""

import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import indexable, column_or_1d
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression


def normalize(score, score_min=-1, score_max=1):
    return (score-score_min)/(score_max-score_min)


# The code below was adapted from https://github.com/betacal/betacal.github.io the original repo for the paper
# Beta regression model with three parameters introduced in
#     Kull, M., Silva Filho, T.M. and Flach, P. Beta calibration: a well-founded
#     and easily implemented improvement on logistic calibration for binary
#     classifiers. AISTATS 2017.

def _beta_calibration(df, y, sample_weight=None):

    df = column_or_1d(df).reshape(-1, 1)
    eps = np.finfo(df.dtype).eps
    df = np.clip(df, eps, 1-eps)
    y = column_or_1d(y)

    x = np.hstack((df, 1. - df))
    x = np.log(x)
    x[:, 1] *= -1

    lr = LogisticRegression(C=99999999999)
    lr.fit(x, y, sample_weight)
    coefs = lr.coef_[0]

    if coefs[0] < 0:
        x = x[:, 1].reshape(-1, 1)
        lr = LogisticRegression(C=99999999999)
        lr.fit(x, y, sample_weight)
        coefs = lr.coef_[0]
        a = 0
        b = coefs[0]
    elif coefs[1] < 0:
        x = x[:, 0].reshape(-1, 1)
        lr = LogisticRegression(C=99999999999)
        lr.fit(x, y, sample_weight)
        coefs = lr.coef_[0]
        a = coefs[0]
        b = 0
    else:
        a = coefs[0]
        b = coefs[1]
    inter = lr.intercept_[0]

    m = minimize_scalar(lambda mh: np.abs(b*np.log(1.-mh)-a*np.log(mh)-inter),
                        bounds=[0, 1], method='Bounded').x
    map = [a, b, m]
    return map, lr


class BetaCalibration(BaseEstimator, RegressorMixin):
    """Beta regression model with three parameters introduced in
    Kull, M., Silva Filho, T.M. and Flach, P. Beta calibration: a well-founded
    and easily implemented improvement on logistic calibration for binary
    classifiers. AISTATS 2017.
    Attributes
    ----------
    map_ : array-like, shape (3,)
        Array containing the coefficients of the model (a and b) and the
        midpoint m. Takes the form map_ = [a, b, m]
    lr_ : sklearn.linear_model.LogisticRegression
        Internal logistic regression used to train the model.
    """

    def __init__(self, scores, ground_truth, score_min=-1, score_max=1):
        self.score_min = score_min
        self.score_max = score_max
        X = column_or_1d(normalize(scores, score_min=score_min, score_max=score_max))
        y = column_or_1d(ground_truth)
        X, y = indexable(X, y)
        self.map_, self.lr_ = _beta_calibration(X, y, )

    def predict(self, S):
        """Predict new values.
        Parameters
        ----------
        S : array-like, shape (n_samples,)
            Data to predict from.
        Returns
        -------
        S_ : array, shape (n_samples,)
            The predicted values.
        """
        df = column_or_1d(normalize(S, score_min=self.score_min, score_max=self.score_max)).reshape(-1, 1)
        eps = np.finfo(df.dtype).eps
        df = np.clip(df, eps, 1 - eps)

        x = np.hstack((df, 1. - df))
        x = np.log(x)
        x[:, 1] *= -1
        if self.map_[0] == 0:
            x = x[:, 1].reshape(-1, 1)
        elif self.map_[1] == 0:
            x = x[:, 0].reshape(-1, 1)

        return self.lr_.predict_proba(x)[:, 1]
