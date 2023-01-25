from sklearn.cluster import KMeans
import numpy as np

from argparse import Namespace
from dataset import Dataset

from calibrationMethods import BetaCalibration

def faircal(dataset: Dataset, conf: Namespace) -> np.ndarray:
    """
    Run the FairCal algorithm. It uses KMeans to set up a number of clusters,
    it then sets up a BetaCalibration instance on each cluster, which is used
    to make a prediction on image pairs.

    If the image pair is split across clusters (ie image1 in cluster i and
    image2 in cluster j, with i != j), take the weighted average of calibrated
    scores using calibrator i and j, weighed by the cluster sizes of i and j.

    Parameters:
        dataset: Dataset - A dataset instance
        conf: Namespace - Configuration of current approach

    Returns:
        score: np.ndarray - Score of used approach
    """
    # Set up dataframe from dataset
    df = dataset.df.copy()
    df['test'] = df['fold'] == dataset.fold
    df['score'] = df[dataset.feature]

    # Get KMeans instance from dataset for the provided number of clusters
    print("Training using KMeans...")
    kmeans: KMeans = dataset.train_cluster(n_clusters=conf.n_cluster, save=True)

    # Get embeddings from dataset, together with mappers
    print("Predicting using KMeans...")
    embeddings, idx2path = dataset.get_embeddings(train=None, return_mapper=True)
    path2embidx = dict((path,i) for i,path in enumerate(idx2path))

    # Predict on all clusters
    cluster_assignments = kmeans.predict(embeddings)

    # Save cluster assignment for each pair
    df['cluster1'] = df['path1'].apply(lambda path: cluster_assignments[ path2embidx[path] ] )
    df['cluster2'] = df['path2'].apply(lambda path: cluster_assignments[ path2embidx[path] ] )

    # Set up calibrators for each of the clusters
    print("Setting up calibrators for each cluster...")
    calibrators = [None]*conf.n_cluster
    cluster_sizes = np.zeros(conf.n_cluster, dtype='int')

    for cluster in range(conf.n_cluster):

        # Now select the train image pairs where both images are assigned to the current cluster
        select = ((df['cluster1'] == cluster) | (df['cluster2'] == cluster)) & (df['test'] == False)

        # Take note of the cluster size (is used for weighted average)
        cluster_sizes[cluster] = np.count_nonzero(select)

        # Set up calibrator on current selection
        calibrator = BetaCalibration(df['score'][select].astype(float), df['same'][select].astype(int), score_min=-1, score_max=1)
        calibrators[cluster] = calibrator

    # Set up calibrated scores as all zeros
    calibrated_score = np.zeros_like(df['score'], dtype='float')

    # Normalizing factor, to take a weighted average of calibrated scores
    norm_fact = np.zeros_like(df['score'])

    # Iterate clusters again
    # TODO is this second loop necessary? Can be combined?
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

    # Normalize calibrated scores by the cluster sizes
    return calibrated_score/norm_fact
