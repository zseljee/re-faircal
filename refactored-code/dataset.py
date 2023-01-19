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
        self.name = dataset
        
        # Load dataset constants
        self.consts: dict[str, Any] = AVAILABLE_DATASETS[dataset]
        
        # Read CSV of pairs
        self._df = pd.read_csv( self.consts['csv'] )
        self.df = self._df.copy()
        self.folds = self._df['fold'].unique()

        # Load embeddings
        self.load_embeddings(self.consts['embeddings'], feature)
    

    def iterate_subgroups(self) -> Iterable[ dict[str, str] ]:
        """
        Yield each combination of sensitive attributes for current selection

        For exmaple: if 'Gender' is only sensitive attribute
        with column g1 and g2 for the pair of images, this method will yield
        {'g1': 'Female', 'g2': 'Female'}
        {'g1': 'Female', 'g2': 'Male'}
        {'g1': 'Male', 'g2': 'Female'}
        {'g1': 'Male', 'g2': 'Male'}

        Resulting dictionary can be readily used in Dataset.select!
        """

        # For each column, save mapping to unique values in that column
        attributes: dict[str, set] = dict()
        
        # Fill 'attributes' by iterating sensitive attributes
        for attribute, columns in self.consts['sensitive_attributes'].items():
            attribute: str
            columns: tuple[str, str] # column of left and right item of the face pair

            for col in columns:
                # For this column, add set of possibilities
                attributes[col] = set( self.df[col] )
        
        # Now combine each value in each column with each other
        # ie (F,F), (F,M), (M,F), (M,M) for gender
        for combination in itertools.product(*attributes.values()):

            # Yield dict that maps column name to value in that column
            yield dict(zip(attributes.keys(), combination))
    

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