import os
import pickle
import time
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve

from calibration_methods import (BetaCalibration, BinningCalibration,
                                 IsotonicCalibration)


FeatureType: TypeAlias = Literal["facenet", "facenet-webface", "arcface"]


def baseline(scores, ground_truth, nbins, calibration_method, score_min=-1, score_max=1):
    if calibration_method == 'binning':
        calibration = BinningCalibration(scores['cal'], ground_truth['cal'], score_min=score_min, score_max=score_max,
                                         nbins=nbins)
    elif calibration_method == 'isotonic_regression':
        calibration = IsotonicCalibration(scores['cal'], ground_truth['cal'], score_min=score_min, score_max=score_max)
    elif calibration_method == 'beta':
        calibration = BetaCalibration(scores['cal'], ground_truth['cal'], score_min=score_min, score_max=score_max)
    else:
        raise ValueError('Calibration method %s not available' % calibration_method)
    confidences = {'cal': calibration.predict(scores['cal']), 'test': calibration.predict(scores['test'])}
    return confidences


def oracle(scores, ground_truth, subgroup_scores, subgroups, nbins, calibration_method):
    confidences = {}
    for dataset in ['cal', 'test']:
        confidences[dataset] = {}
        for att in subgroups.keys():
            confidences[dataset][att] = np.zeros(len(scores[dataset]))

    for att in subgroups.keys():
        for subgroup in subgroups[att]:
            select = {}
            for dataset in ['cal', 'test']:
                select[dataset] = np.logical_and(
                    subgroup_scores[dataset][att]['left'] == subgroup,
                    subgroup_scores[dataset][att]['right'] == subgroup
                )

            scores_cal_subgroup = scores['cal'][select['cal']]
            ground_truth_cal_subgroup = ground_truth['cal'][select['cal']]
            if calibration_method == 'binning':
                calibration = BinningCalibration(scores_cal_subgroup, ground_truth_cal_subgroup, nbins=nbins)
            elif calibration_method == 'isotonic_regression':
                calibration = IsotonicCalibration(scores_cal_subgroup, ground_truth_cal_subgroup)
            elif calibration_method == 'beta':
                calibration = BetaCalibration(scores_cal_subgroup, ground_truth_cal_subgroup)
            else:
                raise ValueError('Calibration method %s not available' % calibration_method)

            confidences['cal'][att][select['cal']] = calibration.predict(scores_cal_subgroup)
            confidences['test'][att][select['test']] = calibration.predict(scores['test'][select['test']])
    return confidences


def cluster_methods(nbins, calibration_method, dataset_name, feature, fold, db_fold, n_clusters,
                    score_normalization, fpr):
    # k-means algorithm
    saveto = f"experiments/kmeans/{dataset_name}_{feature}_nclusters{n_clusters}_fold{fold}.npy"
    if not os.path.exists(saveto):
        np.save(saveto, {})
        embeddings = None
        if dataset_name == 'rfw':
            embeddings = collect_embeddings_rfw(feature, db_fold['cal'])
        elif 'bfw' in dataset_name:
            embeddings = collect_embeddings_bfw(feature, db_fold['cal'])
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(embeddings)
        np.save(saveto, kmeans)
    else:
        while True:
            kmeans = np.load(saveto, allow_pickle=True).item()
            if type(kmeans) != dict:
                break
            print('Waiting for KMeans to be computed')
            time.sleep(60)
    if dataset_name == 'rfw':
        r = collect_miscellania_rfw(n_clusters, feature, kmeans, db_fold)
    elif 'bfw' in dataset_name:
        r = collect_miscellania_bfw(n_clusters, feature, kmeans, db_fold)
    else:
        raise ValueError('Dataset %s not available' % dataset_name)
    scores = r[0]
    ground_truth = r[1]
    clusters = r[2]
    cluster_scores = r[3]

    print('Statistics Cluster K = %d' % n_clusters)
    stats = np.zeros(n_clusters)
    for i_cluster in range(n_clusters):
        select = np.logical_or(cluster_scores['cal'][:, 0] == i_cluster, cluster_scores['cal'][:, 1] == i_cluster)
        clusters[i_cluster]['scores']['cal'] = scores['cal'][select]
        clusters[i_cluster]['ground_truth']['cal'] = ground_truth['cal'][select]
        stats[i_cluster] = len(clusters[i_cluster]['scores']['cal'])

    print('Minimum number of pairs in clusters %d' % (min(stats)))
    print('Maximum number of pairs in clusters %d' % (max(stats)))
    print('Median number of pairs in clusters %1.1f' % (np.median(stats)))
    print('Mean number of pairs in clusters %1.1f' % (np.mean(stats)))

    if score_normalization:

        global_threshold = find_threshold(scores['cal'], ground_truth['cal'], fpr)
        local_threshold = np.zeros(n_clusters)

        for i_cluster in range(n_clusters):
            scores_cal = clusters[i_cluster]['scores']['cal']
            ground_truth_cal = clusters[i_cluster]['ground_truth']['cal']
            local_threshold[i_cluster] = find_threshold(scores_cal, ground_truth_cal, fpr)

        fair_scores = {}

        for dataset in ['cal', 'test']:
            fair_scores[dataset] = np.zeros(len(scores[dataset]))
            for i_cluster in range(n_clusters):
                for t in [0, 1]:
                    select = cluster_scores[dataset][:, t] == i_cluster
                    fair_scores[dataset][select] += local_threshold[i_cluster] - global_threshold

            fair_scores[dataset] = scores[dataset] - fair_scores[dataset] / 2

        # The fair scores are no longer cosine similarity scores so they may not lie in the interval [-1,1]
        fair_scores_max = 1 - min(local_threshold - global_threshold)
        fair_scores_min = -1 - max(local_threshold - global_threshold)

        confidences = baseline(
            fair_scores,
            ground_truth,
            nbins,
            calibration_method,
            score_min=fair_scores_min,
            score_max=fair_scores_max
        )
    else:
        fair_scores = {}
        confidences = {}

        # Fit calibration
        cluster_calibration_method = {}
        for i_cluster in range(n_clusters):
            scores_cal = clusters[i_cluster]['scores']['cal']
            ground_truth_cal = clusters[i_cluster]['ground_truth']['cal']
            if calibration_method == 'binning':
                cluster_calibration_method[i_cluster] = BinningCalibration(scores_cal, ground_truth_cal, nbins=nbins)
            elif calibration_method == 'isotonic_regression':
                cluster_calibration_method[i_cluster] = IsotonicCalibration(scores_cal, ground_truth_cal)
            elif calibration_method == 'beta':
                cluster_calibration_method[i_cluster] = BetaCalibration(scores_cal, ground_truth_cal)
            clusters[i_cluster]['confidences'] = {}
            clusters[i_cluster]['confidences']['cal'] = cluster_calibration_method[i_cluster].predict(scores_cal)

        for dataset in ['cal', 'test']:
            confidences[dataset] = np.zeros(len(scores[dataset]))
            p = np.zeros(len(scores[dataset]))
            for i_cluster in range(n_clusters):
                for t in [0, 1]:
                    select = cluster_scores[dataset][:, t] == i_cluster
                    aux = scores[dataset][select]
                    if len(aux) > 0:
                        aux = cluster_calibration_method[i_cluster].predict(aux)
                        confidences[dataset][select] += aux * stats[i_cluster]
                        p[select] += stats[i_cluster]
            confidences[dataset] = confidences[dataset] / p

    return scores, ground_truth, confidences, fair_scores


def find_threshold(scores, ground_truth, fpr_threshold):
    far, tar, thresholds = roc_curve(ground_truth, scores, drop_intermediate=True)
    aux = np.abs(far - fpr_threshold)
    return np.min(thresholds[aux == np.min(aux)])


def collect_embeddings_rfw(feature: FeatureType, db_cal: pd.DataFrame):
    """ Create a 2D array of embeddings in the dataframe used for calibrating
    the K-means algorithm.
    """
    # collect embeddings of all the images in the calibration set
    embeddings = []  # all embeddings are in a 512-dimensional space
    if feature != 'arcface':
        for subgroup in ['African', 'Asian', 'Caucasian', 'Indian']:
            subgroup_embeddings: dict[str, NDArray] = pickle.load(open('data/rfw/' + subgroup + '_' + feature + '_embeddings.pickle', 'rb'))
            select = db_cal['ethnicity'] == subgroup

            # Create a single dataframe of all unique faces + photos that are used.
            unique_faces_subgroup = pd.concat([
                db_cal[select][["id1", "num1"]]\
                    .rename({"id1": "id", "num1": "num"}),
                db_cal[select][["id2", "num2"]]\
                    .rename({"id2": "id", "num2": "num"}),
            ])\
                .groupby(["id", "num"])\
                .first().reset_index()\
                .rename({"id": "foldername"})

            unique_faces_subgroup["filename"] = unique_faces_subgroup["foldername"] + '_000' + unique_faces_subgroup["num"].astype(str) + ".jpg"

            # Add all unique embeddings
            for i, row in unique_faces_subgroup.iterrows():
                folder_name, file_name = row["foldername"], row["filename"]
                key = 'rfw/data/' + subgroup + '_cropped/' + folder_name + '/' + file_name
                embeddings.append(subgroup_embeddings[key].reshape(1, -1))

    else:  # facenet, facenet-webface
        all_embeddings: dict[str, NDArray] = pickle.load(open('data/rfw/rfw_' + feature + '_embeddings.pickle', 'rb'))
        for subgroup in ['African', 'Asian', 'Caucasian', 'Indian']:
            select = db_cal['ethnicity'] == subgroup

            # Create a single dataframe of all unique faces + photos that are used.
            unique_faces_subgroup = pd.concat([
                db_cal[select][["id1", "num1"]]\
                    .rename({"id1": "id", "num1": "num"}),
                db_cal[select][["id2", "num2"]]\
                    .rename({"id2": "id", "num2": "num"}),
            ])\
                .groupby(["id", "num"])\
                .first().reset_index()\
                .rename({"id": "foldername"})

            unique_faces_subgroup["filename"] = unique_faces_subgroup["foldername"] + '_000' + unique_faces_subgroup["num"].astype(str) + ".jpg"

            # Add all unique embeddings
            for i, row in unique_faces_subgroup.iterrows():
                folder_name, file_name = row["foldername"], row["filename"]
                key = 'rfw/data/' + subgroup + '/' + folder_name + '/' + file_name
                embeddings.append(all_embeddings[key].reshape(1, -1))

    return np.concatenate(embeddings)


def collect_embeddings_bfw(feature: FeatureType, db_cal: pd.DataFrame):
    """ Create a 2D array of embeddings in the dataframe used for calibrating
    the K-means algorithm.
    """
    all_embeddings = pickle.load(open('data/bfw/' + feature + '_embeddings.pickle', 'rb'))
    unique_file_names = np.union1d(db_cal["path1"].unique(), db_cal["path2"].unique())
    embeddings = np.concatenate([
        all_embeddings[filename].reshape(1, -1) for filename in unique_file_names
    ])

    return embeddings


def collect_miscellania_rfw(n_clusters, feature, kmeans, db_fold):
    # setup clusters
    clusters = {}
    for i_cluster in range(n_clusters):
        clusters[i_cluster] = {}

        for variable in ['scores', 'ground_truth']:
            clusters[i_cluster][variable] = {}
            for dataset in ['cal', 'test']:
                clusters[i_cluster][variable][dataset] = []
    scores = {}
    ground_truth = {}
    cluster_scores = {}
    for dataset in ['cal', 'test']:
        scores[dataset] = np.zeros(len(db_fold[dataset]))
        ground_truth[dataset] = np.zeros(len(db_fold[dataset])).astype(bool)
        cluster_scores[dataset] = np.zeros((len(db_fold[dataset]), 2)).astype(int)

    # collect scores, ground_truth per cluster for the calibration set

    if feature != 'arcface':
        subgroup_old = ''
        temp = None
        for dataset, db in zip(['cal', 'test'], [db_fold['cal'], db_fold['test']]):
            scores[dataset] = np.array(db[feature])
            ground_truth[dataset] = np.array(db['same'].astype(bool))
            for i in range(len(db)):
                subgroup = db['ethnicity'].iloc[i]
                if subgroup != subgroup_old:
                    temp = pickle.load(
                        open('data/rfw/' + subgroup + '_' + feature + '_embeddings.pickle', 'rb'))
                subgroup_old = subgroup

                t = 0
                for id_face, num_face in zip(['id1', 'id2'], ['num1', 'num2']):
                    folder_name = db[id_face].iloc[i]
                    file_name = db[id_face].iloc[i] + '_000' + str(db[num_face].iloc[i]) + '.jpg'
                    key = 'rfw/data/' + subgroup + '_cropped/' + folder_name + '/' + file_name
                    i_cluster = kmeans.predict(temp[key])[0]
                    cluster_scores[dataset][i, t] = i_cluster
                    t += 1
    else:
        temp = pickle.load(open('data/rfw/rfw_' + feature + '_embeddings.pickle', 'rb'))
        for dataset, db in zip(['cal', 'test'], [db_fold['cal'], db_fold['test']]):
            scores[dataset] = np.array(db[feature])
            ground_truth[dataset] = np.array(db['same'].astype(bool))
            for i in range(len(db)):
                subgroup = db['ethnicity'].iloc[i]
                t = 0
                for id_face, num_face in zip(['id1', 'id2'], ['num1', 'num2']):
                    folder_name = db[id_face].iloc[i]
                    file_name = db[id_face].iloc[i] + '_000' + str(db[num_face].iloc[i]) + '.jpg'
                    key = 'rfw/data/' + subgroup + '/' + folder_name + '/' + file_name
                    i_cluster = kmeans.predict(temp[key].reshape(1, -1).astype(float))[0]
                    cluster_scores[dataset][i, t] = i_cluster
                    t += 1

    return scores, ground_truth, clusters, cluster_scores


def collect_miscellania_bfw(n_clusters, feature, kmeans, db_fold):
    # setup clusters
    clusters = {}
    for i_cluster in range(n_clusters):
        clusters[i_cluster] = {}

        for variable in ['scores', 'ground_truth']:
            clusters[i_cluster][variable] = {}
            for dataset in ['cal', 'test']:
                clusters[i_cluster][variable][dataset] = []
    scores = {}
    ground_truth = {}
    cluster_scores = {}
    for dataset in ['cal', 'test']:
        scores[dataset] = np.zeros(len(db_fold[dataset]))
        ground_truth[dataset] = np.zeros(len(db_fold[dataset])).astype(bool)
        cluster_scores[dataset] = np.zeros((len(db_fold[dataset]), 2)).astype(int)

    # collect scores and ground_truth per cluster for the calibration set
    temp = pickle.load(open('data/bfw/' + feature + '_embeddings.pickle', 'rb'))
    for dataset, db in zip(['cal', 'test'], [db_fold['cal'], db_fold['test']]):
        scores[dataset] = np.array(db[feature])
        ground_truth[dataset] = np.array(db['same'].astype(bool))
        for i in range(len(db)):
            t = 0
            for path in ['path1', 'path2']:
                key = db[path].iloc[i]
                if feature == 'arcface':
                    i_cluster = kmeans.predict(temp[key].reshape(1, -1).astype(float))[0]
                else:
                    i_cluster = kmeans.predict(temp[key])[0]

                cluster_scores[dataset][i, t] = i_cluster
                t += 1

    return scores, ground_truth, clusters, cluster_scores
