from sklearn.cluster import KMeans
import numpy as np

from argparse import Namespace
from dataset import Dataset

from approaches.utils import get_threshold
from calibrationMethods import BetaCalibration

def faircal(dataset: Dataset, conf: Namespace) -> dict:
    data = {
        'confidendences': {},
        'threshold': {},
        'fpr': {}
    }

    dataset.select(None)

    kmeans: KMeans = dataset.cluster(n_clusters=conf.n_cluster)

    embeddings, idx2path = dataset.get_embeddings(train=True, return_mapper=True)
    path2idx = dict((path, idx) for idx,path in enumerate(idx2path))
    cluster_assignment = kmeans.predict(embeddings)

    # Set up calibrator on entire train set
    scores_train, gt_train = dataset.get_scores(train=True, include_gt=True)
    calibrator = BetaCalibration(scores_train, gt_train)

    # Use calibrated scores to set a threshold
    calibrated_scores = calibrator.predict( scores_train )
    thr = get_threshold(calibrated_scores, gt_train, conf.fpr_thr)

    # Save calibrated scores on test set too
    scores_test, gt_test = dataset.get_scores(train=False, include_gt=True)

    confidences = {
        'cal': calibrated_scores,
        'test': calibrator.predict( scores_test ),
    }

    fpr = 0. # TODO compute FPR for test set

    data['confidendences'][f'global'] = confidences
    data['threshold'][f'global'] = thr
    data['fpr'][f'global'] = fpr


    # Set up calibrators for each of the clusters
    calibrators = [None]*conf.n_cluster
    cluster_sizes = [0]*conf.n_cluster

    df = dataset.df.copy()
    
    scores = df[dataset.feature]
    ground_truth = df['same']
    calibrated_scores = np.zeros_like(scores)
    norm_fact = np.zeros_like(scores)

    for cluster in range(conf.n_cluster):

        # Select the paths where the embedding is assigned to the current cluster
        paths = idx2path[ (cluster_assignment == cluster) ]

        # Select data where both images are assigned to the current cluster
        dataset.select(path1=paths, path2=paths)
        
        cluster_sizes[cluster] = len(dataset)

        # Set up calibrator on current selection
        scores_train, gt_train = dataset.get_scores(train=True, include_gt=True)
        calibrator = BetaCalibration(scores_train, gt_train, score_min=-1, score_max=1)
        calibrators[cluster] = calibrator

        select = (df['path1'].isin(paths))
        calibrated_scores[ select ] += calibrator.predict( scores[select] ) * cluster_sizes[cluster]
        norm_fact[ select ] += cluster_sizes[cluster]

        select = (df['path2'].isin(paths))
        calibrated_scores[ select ] += calibrator.predict( scores[select] ) * cluster_sizes[cluster]
        norm_fact[ select ] += cluster_sizes[cluster]
    
    calibrated_scores /= norm_fact

    data['confidences']['cal'] = calibrated_scores[ df['fold'] != dataset.fold ]
    data['confidences']['test'] = calibrated_scores[ df['fold'] == dataset.fold ]

    # TODO very much the wrong way to do this
    for subgroup in dataset.iterate_subgroups(use_attributes='ethinicty'):
        select = (df['e1'] == subgroup['ethnicity']) & (df['e2'] == subgroup['ethnicity'])

        thr = get_threshold(calibrated_scores[select], ground_truth[select], conf.fpr_thr)
        fpr = 0. # TODO

        data['threshold'][subgroup['ethnicity']] = thr
        data['fpr'][subgroup['ethnicity']] = fpr

    return data