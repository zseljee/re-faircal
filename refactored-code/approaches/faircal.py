from sklearn.cluster import KMeans
import numpy as np

from argparse import Namespace
from dataset import Dataset

from approaches.utils import get_threshold
from calibrationMethods import BetaCalibration

def faircal(dataset: Dataset, conf: Namespace) -> dict:
    data = {'confidences': dict(),
            'threshold': dict(),
            'fpr': dict()
           }

    dataset.select(None)

    print("Training using KMeans...")
    kmeans: KMeans = dataset.train_cluster(n_clusters=conf.n_cluster, save=True)

    print("Predicting using KMeans...")
    # TODO
    embeddings = np.copy(dataset._embeddings)
    idx2path = np.array(dataset.embidx2path)
    cluster_assignment = kmeans.predict(embeddings)

    # Set up calibrator on entire train set
    print("Setting up global calibrator...")
    scores_train, gt_train = dataset.get_scores(train=True, include_gt=True)
    calibrator = BetaCalibration(scores_train, gt_train)

    # Use calibrated scores to set a threshold
    calibrated_scores = calibrator.predict( scores_train )
    thr = get_threshold(calibrated_scores, gt_train, conf.fpr_thr)

    # Save calibrated scores on test set too
    scores_test, gt_test = dataset.get_scores(train=False, include_gt=True)
    fpr = 0. # TODO compute FPR for test set

    data['threshold'][f'global'] = thr
    data['fpr'][f'global'] = fpr


    # Set up calibrators for each of the clusters
    print("Setting up calibrators for each cluster...")
    calibrators = [None]*conf.n_cluster
    cluster_sizes = np.zeros(conf.n_cluster, dtype='int')

    df = dataset.df.copy()
    
    scores = df[dataset.feature]
    ground_truth = df['same']
    calibrated_scores = np.zeros_like(scores)

    for cluster in range(conf.n_cluster):

        # Get the paths where the embedding is assigned to the current cluster
        paths = set( idx2path[ (cluster_assignment == cluster) ] )

        # Now select the image pairs where both images are assigned to the current cluster
        select = (df['path1'].isin(paths)) & (df['path2'].isin(paths))
        
        # Take note of the cluster size (is used below)
        cluster_sizes[cluster] = np.sum(select)

        # Set up calibrator on current selection
        calibrator = BetaCalibration(scores[select], ground_truth[select], score_min=-1, score_max=1)
        calibrators[cluster] = calibrator
    
    # Normalizing factor, to take a weighted average of calibrated scores
    norm_fact = np.zeros_like(scores)

    # Iterate clusters again
    for cluster in range(conf.n_cluster):

        # Select the paths where the embedding is assigned to the current cluster
        paths = set( idx2path[ (cluster_assignment == cluster) ] )

        # Select image pairs where the left image is assigned to the current cluster
        select = (df['path1'].isin(paths))
        # Calibrate score using calibrator of current cluster, weighed by the cluster size
        calibrated_scores[ select ] += calibrators[cluster].predict( scores[select] ) * cluster_sizes[cluster]
        # Add the cluster size to current selection of pairs, to take a weighted average of calibrated scores
        norm_fact[ select ] += cluster_sizes[cluster]

        # Select image pairs where the right image is assigned to the current cluster
        select = (df['path2'].isin(paths))
        # Calibrate score using calibrator of current cluster, weighed by the cluster size
        calibrated_scores[ select ] += calibrators[cluster].predict( scores[select] ) * cluster_sizes[cluster]
        # Add the cluster size to current selection of pairs, to take a weighted average of calibrated scores
        norm_fact[ select ] += cluster_sizes[cluster]

    # Normalize calibrated scores by cumulative cluster size
    calibrated_scores /= norm_fact

    # Save confidences
    data['confidences'] = {
        'cal': calibrated_scores[ df['fold'] != dataset.fold ],
        'test': calibrated_scores[ df['fold'] == dataset.fold ],
    }

    print("Computing results for subgroups...")
    for subgroup in dataset.iterate_subgroups(use_attributes='ethnicity'):
        select = (df['ethnicity'] == subgroup['ethnicity']) #& (df['e2'] == subgroup['ethnicity'])

        thr = get_threshold(calibrated_scores[select], ground_truth[select], conf.fpr_thr)
        fpr = 0. # TODO

        data['threshold'][subgroup['ethnicity']] = thr
        data['fpr'][subgroup['ethnicity']] = fpr

    return data