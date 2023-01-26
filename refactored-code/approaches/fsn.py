from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve
import numpy as np

from argparse import Namespace
from dataset import Dataset

from calibrationMethods import BetaCalibration
from .utils import thr_at_fpr_from_score

def fsn(dataset: Dataset, conf: Namespace) -> np.ndarray:
    """
    TODO

    Parameters:
        dataset: Dataset - A dataset instance
        conf: Namespace - Configuration of current approach

    Returns:
        score: np.ndarray - Score of used approach
    """
    # Set up dataframe from dataset
    df = dataset.df.copy()
    df['test'] = (df['fold'] == dataset.fold)
    df['score'] = df[dataset.feature]

    score_cal = df['score'][df['test'] == False].to_numpy(dtype='float')
    gt_cal = df['same'][df['test'] == False].to_numpy(dtype='int')

    # Get KMeans instance from dataset for the provided number of clusters
    print("Training using KMeans...")
    kmeans: KMeans = dataset.train_cluster(n_clusters=conf.n_cluster, save=True)

    # Get embeddings from dataset, together with mapper
    print("Predicting using KMeans...")
    embeddings, idx2path = dataset.get_embeddings(train=None, return_mapper=True)
    path2embidx = dict((path,i) for i,path in enumerate(idx2path))

    # Predict on all clusters
    cluster_assignments = kmeans.predict(embeddings)

    # Save cluster assignment for each pair
    df['cluster1'] = df['path1'].apply(lambda path: cluster_assignments[ path2embidx[path] ] )
    df['cluster2'] = df['path2'].apply(lambda path: cluster_assignments[ path2embidx[path] ] )

    cluster1_cal = df['cluster1'][ df['test'] == False ]
    cluster2_cal = df['cluster2'][ df['test'] == False ]

    # Find the threshold at which the FPR is the pre-defined target FPR
    global_thr = thr_at_fpr_from_score(score_cal, gt_cal, conf.fpr_thr)

    cluster_thr = np.zeros(conf.n_cluster)
    for i_cluster in range(conf.n_cluster):
        select = (cluster1_cal == i_cluster) | (cluster2_cal == i_cluster)
        cluster_thr[i_cluster] = thr_at_fpr_from_score(score_cal[select], gt_cal[select], conf.fpr_thr)

    calibrated_score = np.zeros_like(df['score'], dtype='float')
    for i_cluster in range(conf.n_cluster):
        select = (df['cluster1'] == i_cluster)
        calibrated_score[select] += cluster_thr[i_cluster] - global_thr

    calibrated_score = df['score'] - (calibrated_score / 2.)

    score_max = 1. - np.min(cluster_thr - global_thr)
    score_min = -1. - np.max(cluster_thr - global_thr)

    calibrator = BetaCalibration(
        scores=calibrated_score,
        ground_truth=df['same'].to_numpy(dtype='int'),
        score_max=score_max,
        score_min=score_min,
    )

    return calibrator.predict(df['score'])
