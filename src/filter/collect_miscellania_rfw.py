import typing
import pandas as pd
import numpy as np
import itertools

def collect_miscellania(ds:"Dataset", kmeans:any, model_type:str, test_fold:int):
    """
    function description


    inputs
        ds: dataset class from dataset.py. 
            The dataset class containts the information about the picture
            and it containts the embeddings of these pictures.
        kmeans: kmeans model trained on the train data
        model_type: string containtin the model that generated the embeddings.
        test_fold: int that indicates the fold of the data that is used for testing purposes

    output:
        model_type: dictionary with the model type
        ground_truth: dictionary with the boolean value about imposter or good pair
        cluster_scores: dictionary with the cluster assignment per item in each subgroup
    """
    model_type = {}
    ground_truth = {}
    cluster_scores = {}

    ds.set_fold(test_fold)

    # for each subgroup loop over calibration and test data
    for subgroup in ds.iterate_subgroups():
        ground_truth[subgroup['ethnicity']] = np.array(ds.df['same'].astype(bool))
        model_type[subgroup['ethnicity']] = np.array(ds.df[model_type])
        ds.select_subgroup(**subgroup)

        # get list of embeddings from the subset_df
        embeddings_train = ds.get_embeddings(False)
        embeddings_test = ds.get_embeddings(True)
        
        # get predictions
        train_predictions = kmeans.predict(embeddings_train)[0]
        test_embeddings = kmeans.predict(embeddings_test)[0]
        
        # for each item in the subgroup predict the kmeans cluster
        cluster_scores[subgroup['ethnicity']] = {'cal': train_predictions, 'test':test_embeddings}

    return model_type, ground_truth, cluster_scores