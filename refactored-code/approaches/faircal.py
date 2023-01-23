from sklearn.cluster import KMeans
import numpy as np

from argparse import Namespace
from dataset import Dataset

from approaches.utils import get_threshold
from calibrationMethods import BetaCalibration

def faircal(dataset: Dataset, conf: Namespace) -> dict:
    data = {'df': dict(),
            'threshold': dict(),
            'fpr': dict()
           }

    dataset.select(None)
    
    df = dataset.df.copy()
    select_test = df['fold'] == dataset.fold
    df['test'] = select_test
    df['score'] = df[dataset.feature]
    df['calibrated_score'] = 0.

    print("Training using KMeans...")
    kmeans: KMeans = dataset.train_cluster(n_clusters=conf.n_cluster, save=True)

    print("Predicting using KMeans...")
    # TODO
    embeddings = np.copy(dataset._embeddings)
    embidx2path = np.array( dataset.embidx2path )
    path2embidx = dataset.path2embidx

    cluster_assignments = kmeans.predict(embeddings)

    df['cluster1'] = df['path1'].apply(lambda path: cluster_assignments[ path2embidx[path] ] )
    df['cluster2'] = df['path2'].apply(lambda path: cluster_assignments[ path2embidx[path] ] )


    # Set up calibrators for each of the clusters
    print("Setting up calibrators for each cluster...")
    calibrators = [None]*conf.n_cluster
    cluster_sizes = np.zeros(conf.n_cluster, dtype='int')

    for cluster in range(conf.n_cluster):

        # Now select the train image pairs where both images are assigned to the current cluster
        select = (df['cluster1'] == cluster) & (df['cluster2'] == cluster) & (~select_test)
        
        # Take note of the cluster size (is used below)
        cluster_sizes[cluster] = np.count_nonzero(select)

        # Set up calibrator on current selection
        calibrator = BetaCalibration(df['score'][select], df['same'][select], score_min=-1, score_max=1)
        calibrators[cluster] = calibrator
    
    # Normalizing factor, to take a weighted average of calibrated scores
    norm_fact = np.zeros_like(df['score'])

    calibrated_score = np.zeros_like(df['score'])

    # Iterate clusters again
    for cluster in range(conf.n_cluster):

        # Select image pairs where the left image is assigned to the current cluster
        select = (df['cluster1'] == cluster)
        # Calibrate score using calibrator of current cluster, weighed by the cluster size
        calibrated_score[ select ] += calibrators[cluster].predict( df['score'][select] ) * cluster_sizes[cluster]
        # Add the cluster size to current selection of pairs, to take a weighted average of calibrated scores
        norm_fact[ select ] += cluster_sizes[cluster]

        # Select image pairs where the right image is assigned to the current cluster
        select = (df['cluster2'] == cluster)
        # Calibrate score using calibrator of current cluster, weighed by the cluster size
        calibrated_score[ select ] += calibrators[cluster].predict( df['score'][select] ) * cluster_sizes[cluster]
        # Add the cluster size to current selection of pairs, to take a weighted average of calibrated scores
        norm_fact[ select ] += cluster_sizes[cluster]

    # Normalize calibrated scores by cumulative cluster size
    df['calibrated_score'] = calibrated_score/norm_fact

    # Use calibrated scores to set a threshold
    print("Computing global results...")
    thr = get_threshold(df['calibrated_score'][~select_test], df['same'][~select_test], conf.fpr_thr)

    # Save calibrated scores on test set too
    fpr = 0. # TODO compute FPR for test set

    data['threshold'][f'global'] = thr
    data['fpr'][f'global'] = fpr

    print("Computing results for subgroups...")
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