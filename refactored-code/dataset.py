from typing import Any, Iterable
import itertools

import pickle
import numpy as np
import pandas as pd

from constants import *

class Dataset(object):
    def __init__(self,
                 dataset: str,
                 feature: str):

        # Check availability of dataset
        if dataset not in AVAILABLE_DATASETS:
            raise ValueError(f"Unkown dataset {dataset}, please choose one of {set(AVAILABLE_DATASETS.keys())}")
        self.name: str = dataset
        
        # Load dataset constants
        self.consts: dict[str, Any] = AVAILABLE_DATASETS[dataset]
        
        # Read CSV of pairs
        self._df: pd.DataFrame = pd.read_csv( self.consts['csv'] )
        self.df: pd.DataFrame = self._df.copy()
        self.folds: np.ndarray = self._df['fold'].unique()
        self.fold: int|None = None

        # Load embeddings
        self.load_embeddings(self.consts['embeddings'], feature)


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
        
        emb_size = shape[0]

        # Set up embeddings matrix
        self._embeddings = np.zeros( (len(self.embidx2path), emb_size) )

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

        Parameters:
            k: int - Fold to choose
        """
        if k is not None and k not in self.folds:
            raise ValueError(f"Unkown fold {k}, please choose from {self.folds}")
        self.fold = k


    def scores(self, approach: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the 'score' and ground truth of the dataset for a given feature as numpy arrays.
        Assumes score is saved in a column in the dataframe with the same name as the approach

        Score is a value between -1 and 1, corresponding to the dot product between the embeddings of the images
        of that pair of images. Should be saved in the csv of the dataset named after the approach that created
        the embeddings, such as 'facenet-webface' or 'arcface'.

        Ground truth is the 'same' column of t
        """
        if approach not in self.df.columns:
            raise ValueError(f"Approach {approach} has no scores in dataset {self.name}. Please add a column with scores for this feature.")
        
        scores = self.df[approach].to_numpy(copy=True)
        ground_truth = self.df['same'].to_numpy(copy=True)#, dtype=int) TODO: current is bool, should be int?
        return scores, ground_truth


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
        
        # Now combine each value for each sensitive attribute with each other
        for combination in itertools.product(*attributes.values()):

            # Yield dict that maps column name to value in that column
            yield dict(zip(attributes.keys(), combination))
    

    def select_subgroup(self, **attributes: dict[str, str]) -> None:
        """
        Select a subgroup of the data from the sensitive attributes
        ie `Dataset.select_subgroup(ethnicity='Asian')` will only select data where the ethnicity is 'Asian'
        
        Will raise a ValueError if kwarg is a 'sensitive_attribute' (as defined in constants.py)
        or if the corresponding value cannot be found in that same value.

        Will call `Dataset.select` using the appropriate columns with values provided values.
        Example: Calling `Dataset.select_subgroup(ethnicity='Asian')` will look up in what columns
        'ethnicity' can be found, let that be 'e1', 'e2'. This function will then call
        `Dataset.select(e1='Asian', e2='Asian')` to select the data where the ethnicity is 'Asian'.

        Parameters:
            kwarg: str - Value to consider for sensitive attribute
        """

        # A dictionary mapping column to the value of that column to use
        select: dict[str, list[Any]] = {}

        # Iterate each sensitive attribute and the value it needs to take
        for attribute, value in attributes.items():
            if attribute not in self.consts['sensitive_attributes']:
                raise ValueError(f"Unkown sensitive attribute {attribute} in dataset {self.name}")
            
            if value not in self.consts['sensitive_attributes'][attribute]['values']:
                raise ValueError(f"Sensitive attribute {attribute} in dataset {self.name} does not take on value {value}")

            # In what columns this sensitive attribute can be found; is of size 2
            columns: list[str] = self.consts['sensitive_attributes'][attribute]['cols']

            # Add columns to selection
            for col in columns:
                select[col] = value

        # Now set dictionary of constraints as current selection
        self.select(**select)

    
    def select(self, keep_existing: bool=False, **constraints: dict[str, Any|list[Any]|set[Any]]) -> None:
        """
        Set some constraints on the dataset, such as fold or ethnicity.

        Each kwargs has to correspond to a column, and each parameter either
        contains a value, list or set of values on which to filter that column

        For example:
        Dataset.select(fold={1,2,3,4}, ethnicity='Asian')
        selects the data from folds 1 to 4, where ethnicity is Asian.

        Parameters:
            keep_existing: bool - Whether to reset entire selection, or continue on previous
            kwarg: Any|list[Any]|set[Any] - A value, a list or a set of values
        """
        df = (self.df if keep_existing else self._df).copy()

        constraintCols = set(constraints.keys())
        dataCols = set(df.columns)
        if not all(col in dataCols for col in constraintCols):
            raise ValueError(f"Constraint include column(s) that do not exist in the data. constraint:{constraintCols}, data:{dataCols}")

        # Initialize mask to include all
        mask = np.full(len(df), True)

        # Iterate constraints
        for col, vals in constraints.items():
            # Convert single value to set of values
            if not isinstance(vals, (list, set)):
                vals = {vals,}

            # logical AND mask with mask for current constraint
            mask &= df[col].isin( set(vals) )
        
        # Set current df as subset of full dataset
        self.df = df[mask]
    

    def __len__(self):
        return len(self.df)