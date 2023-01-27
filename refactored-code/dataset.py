# TODO: Combine `iterate_subgroups` and `select_subgroup` into `iterate_select_subgroup`?
import os
from typing import Any, Iterable
import itertools

import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from constants import *

class Dataset(object):
    def __init__(self,
                 name: str,
                 feature: str):

        # Check availability of dataset
        if name not in AVAILABLE_DATASETS:
            raise ValueError(f"Unkown dataset {name}, please choose one of {set(AVAILABLE_DATASETS.keys())}")
        self.name: str = name
        self.feature: str = feature

        # Load dataset constants
        self.consts: dict[str, Any] = AVAILABLE_DATASETS[name]

        # Read CSV of pairs
        self._df: pd.DataFrame = pd.read_csv( self.consts['csv'] )
        self.df: pd.DataFrame = self._df.copy()
        self.folds: np.ndarray = self._df['fold'].unique()
        self.fold: int|None = None

        self.kmeans: KMeans|None = None
        # load kmeans if exists
        files = os.listdir(DATA_FOLDER)
        if 'kmeans.pkl' in files:
            with open(DATA_FOLDER + 'kmeans.pkl', 'rb') as f:
                self.kmeans = pickle.load(f)
                assert isinstance(self.kmeans, KMeans)

        if feature not in self._df.columns:
            s = f"Could not set up dataset {self.name} with feature {self.feature}"
            s+= f", please specify the column {self.feature} as similiarty score between embeddings."
            raise ValueError(s)

        # Load embeddings
        self.load_embeddings(self.consts['embeddings'], feature)

        # Make sure the dataset containts only data that has been embedded
        paths_df = set( self._df['path1'] ) | set( self._df['path2'] )
        paths_emb = set( self.embidx2path )
        assert all( path in paths_emb for path in paths_df ), "Some images in the dataframe have not been embedded! This concerns "+str(paths_df-paths_emb)


    def load_embeddings(self, embeddings_path: str, feature: str) -> None:
        """
        Given an unformatted path to embedding and a feature, load embeddings
        Assumes embedding can be formatted by filling in 'feature' variable

        Assumes the resulting file can be loaded using pickle.
        the resulting data should be a dictionary mapping a path to an embedding
        These paths are saved in the list Dataset.embidx2path and dictionary
        Dataset.path2embidx

        Embeddings are loaded into the np.ndarray Dataset._embeddings
        """

        # Convert unformatted path for feature
        fname = embeddings_path.format(feature)
        if not os.path.isfile(fname):
            raise ValueError("Could not find embeddings for dataset {} using feature {}. Please create {}".format(self.name, feature, fname))

        # Load embeddings dictionary, mapping paths to embeddings
        with open(fname, 'rb') as f:
            embeddings_dict: dict[str, np.ndarray] = pickle.load(f)

        # Instead, map path to an embedding index
        self.embidx2path: list[str] = list(embeddings_dict.keys())
        self.path2embidx: dict[str, int] = dict((path,i) for i,path in enumerate(self.embidx2path))

        # TODO: This can be optimized using np.ravel

        # Deducde embedding shape from first embedding in the dict
        # Squeeze to convert shapes (X,1) or (1,X) to (X,)
        shape: tuple[int] = np.squeeze( next(iter(embeddings_dict.values())) ).shape

        # If, after squeezing, still >1D, I dont know what to do
        if len(shape) != 1:
            raise ValueError("Could not parse embedding of shape", shape)

        self.emb_size = shape[0]

        # Set up embeddings matrix
        self._embeddings = np.zeros( (len(self.embidx2path), self.emb_size) )

        # Fill with embeddings from dict
        for i,path in enumerate(self.embidx2path):
            self._embeddings[i] = np.squeeze( embeddings_dict[path] )


    def set_fold(self, k: int|None):
        """
        Set the current fold of the dataset. Depending on whether testing or training,
        use data where df['fold'] == k or df['fold'] != k resp.
        If k is None, use all folds

        Will raise ValueError if k is not found in df['fold']. You are adviced to use
        for k in Dataset.folds:
            Dataset.set_fold(k)

        Set k to None to include all data

        Parameters:
            k: int|None - Fold to choose
        """
        if k is not None and k not in self.folds:
            raise ValueError(f"Unkown fold {k}, please choose from {self.folds}")
        self.fold = k


    def get_scores(self, include_gt: bool = False, train: bool=False) -> np.ndarray|tuple[np.ndarray, np.ndarray]:
        """
        Returns the 'score' and ground truth of the dataset for a given feature as numpy arrays.
        Assumes score is saved in a column in the dataframe with the same name as the feature

        Score is a value between -1 and 1, corresponding to the dot product between the embeddings of the images
        of that pair of images. Should be saved in the csv of the dataset named after the feature that created
        the embeddings, such as 'facenet-webface' or 'arcface'.

        If include_gt is set to True, ground truth is also returned as numpy array. Ground truth is the
        'same' column of the dataframe.

        If train is set to True, only include scores from the train part of the current fold. When set to false,
        only include data of the current fold. Ignored if fold is not set

        Parameters:
            include_gr: bool - Whether to also return the ground truth
            train: bool - Whether to use train or test data, ignored if fold is not set

        Returns:
            scores: np.ndarray - The scores of the current approach
            ground_truth: np.ndarray - Ground truth, either true or false.
                                       Only included if include_gt is set to True
        """
        # Get scores using set feature
        scores = self.df[self.feature].to_numpy(copy=True)

        # Also return ground truth
        ground_truth = self.df['same'].to_numpy(copy=True)#, dtype=int) TODO: current is bool, should be int?

        # Only take data of the current fold
        if self.fold is not None:
            select = (self.df['fold'] != self.fold) if train else (self.df['fold'] == self.fold)
            scores = scores[select]
            ground_truth = ground_truth[select]

        # Only return scores
        if include_gt:
            return scores, ground_truth
        else:
            return scores


    def get_embeddings(self, train: bool|None=False, return_mapper: bool=False) -> np.ndarray|tuple[np.ndarray, np.ndarray]:
        """
        Get a numpy array containing the embeddings of the dataset, where each embedding is saved as a row
        If a fold is set (ie `Dataset.fold is not None`), use the `train` parameter to choose between the training
        or test data of the current fold. If no fold is set, `train` is ignored.

        Embeddings use the current selection of data, using a different `Dataset.select` will (potentially) result
        in different embeddings. The returned embeddings come from the set of paths of the current dataframe,
        to get embedding pairs, see ... TODO

        Returned embeddings are a copy of the embeddings saved internally.

        Parameters:
            train: bool|None - If a fold is set, whether to include data only using that fold (when `train=False`),
            or anything but (when `train=True`). Ignored if no fold is set or `train=None`.

        Returns:
            embeddings: np.ndarray
            idx2path: np.ndarray - A 1D array containing paths, where the ith path is the path of the ith embedding
        """
        # Use current selection
        df = self.df.copy()

        # Use selection of current fold
        if (self.fold is not None) and (train is not None):
            select = (df['fold'] != self.fold) if train else (df['fold'] == self.fold)
            df = df[select]

        # Get embedding paths from path1 and path2 column in data
        paths = set( df['path1'] ) | set( df['path2'] )

        # Convert paths to indices of embeddings
        idxs = [self.path2embidx[path] for path in paths]
        idx2path = np.array([self.embidx2path[idx] for idx in idxs])

        embeddings = np.copy(self._embeddings[idxs])

        # Copy embeddings at idxs
        if return_mapper:
            return embeddings, idx2path
        return embeddings


    def iterate_subgroups(self, use_attributes: str|Iterable[str]|None = None) -> Iterable[ dict[str, Any] ]:
        """

        """
        # Convert use_attributes to a list of values
        if isinstance(use_attributes, str):
            # String to list
            use_attributes = [use_attributes,]
        elif use_attributes is None:
            # 'None' means include all, so create a copy of sensitive attributes
            use_attributes = self.consts['sensitive_attributes'].keys()

        # Attributes is a list mapping a sensitive attribute (such as 'gender') to the values that attribute takes
        attributes: dict[str, list[Any]] = dict()
        for attribute in self.consts['sensitive_attributes']:
            if attribute in use_attributes:
                attributes[attribute] = self.consts['sensitive_attributes'][attribute]['values']

                for value in attributes[attribute]:
                    yield {attribute: value}

        # Now combine each value for each sensitive attribute with each other
        for combination in itertools.product(*attributes.values()):

            # Yield dict that maps column name to value in that column
            yield dict(zip(attributes.keys(), combination))


    def train_cluster(self, n_clusters:int=100, save=False):
        """
        use kmeans clustering to create clusters of the embeddings
        this function will train a kmeans classifier and return it

        input:
            n_clusters: int, (default 100 as used in the paper)
                the number of clusters in the data
            save: bool, if the model is saved

        output:
            kmeans: KMeans, kmeans classifier that can be used to
                predict clusters for new points or get the labels of training points
        """
        # Set up filename
        fname = os.path.join(EXPERIMENT_FOLDER, 'kmeans', f'{self.name}_{self.feature}_nclusters{n_clusters}_fold{self.fold}.pkl')

        # If already trained using these parameters
        if os.path.isfile(fname):

            # Read pickle file
            with open(fname, 'rb') as f:
                kmeans = pickle.load(f)

        else:

            print(f"Fitting KMeans on dataset {self.name} using feature {self.feature}, #clusters {n_clusters} and fold {self.fold}")
            print(f"Saving to {fname}")

            # get embeddings
            embeddings = self.get_embeddings(train=True)

            # train
            kmeans = KMeans(n_clusters=n_clusters, n_init=10).fit(embeddings)

        # store model
        if save:
            with open(fname, 'wb') as f:
                pickle.dump(kmeans, f)

        return kmeans


    def __len__(self):
        return len(self.df)
