import os
from typing import Optional
import tqdm

import numpy as np
import pandas as pd
import pickle

from skimage import io
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class BFWEmbeddings(Dataset):
    def __init__(self,
                 model: Optional[str] = None,
                 embeddings: Optional[str] = None,
                 data_root: str = './data/bfw',
                 csv_file: str = './data/bfw/bfw.csv',
                 dataset: str = 'full',
                ):
        """
        TODO
        """

        # Convert to global paths for interpretable printing
        data_root = os.path.abspath(data_root)
        csv_file = os.path.abspath(csv_file)

        # Save location of data
        self.data_root = data_root

        # Load presets from `model` parameter
        if model in {'facenet', 'facenet_webface'}:
            embeddings = os.path.join(self.data_root, f'{model}_embeddings.pickle')

        # Unkown value for `model` parameter
        elif model is not None:
            raise ValueError(f"Unkown embedding model {model}")

        # `embeddings` should either be set via the code above, or given as parameter
        if embeddings is None:
            raise ValueError("Either set `model` or `embeddings` parameter")

        # Load embeddings as dict mapping path to np.ndarray
        with open(embeddings, 'rb') as f:
            self.embeddings: dict[str, np.ndarray] = pickle.load(f)

        # Load all image pairs
        df = pd.read_csv( csv_file )

        # Filter embeddings to only include embedded images
        paths = set(self.embeddings.keys())
        mask = (df['path1'].isin(paths)) & (df['path2'].isin(paths))
        df = df[ mask ]

        # Set up some pairs
        self.dataframes = {
            'full': df,
            # Will be set using set_fold at the end of __init__
            'train': pd.DataFrame(data=[]),
            'test': pd.DataFrame(data=[])
        }
        # What dataset to use, has to be one of the keys defined above
        if dataset not in self.dataframes.keys():
            raise ValueError(f"Unkown dataset {dataset}, options: {list(self.dataframes.keys())}")
        self.dataset = dataset

        # Save which folds are available
        self.folds = self.dataframes['full']['fold'].unique()
        self.current_fold = None

        # Load it to first fold
        self.set_fold(self.folds[0])


    def set_fold(self, k: int) -> None:
        """
        TODO
        """
        # Make sure provided fold exists
        if k not in self.folds:
            raise KeyError(f"Fold {k} not found in BFW dataset")

        # Check if all is set already
        if k == self.current_fold:
            return

        self.current_fold = k

        mask = self.dataframes['full']['fold'] == k

        # Test dataframe is for given fold
        self.dataframes['test'] = self.dataframes['full'][ mask ]

        # Trainfolds consist of all other folds
        self.dataframes['train'] = self.dataframes['full'][ ~mask ]


    def train(self):
        self.dataset = 'train'
    def calibrate(self):
        self.train()


    def test(self):
        self.dataset = 'test'


    def __len__(self):
        """Give length of DataSet"""
        return len(self.dataframes[self.dataset])


    def __getitem__(self, idx: any) -> "tuple[ np.ndarray, np.ndarray, int, dict[str, any] ]":
        """
        Given some index, give embeddings and label corresponding to that pair
        TODO: Also return persion ID and sensitive attributes?

        Parameters:
            idx: Any - Index of the sample, ie what is passed to BFWEmbeddings[ ... ]

        Returns:
            WIP
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Use iloc if index is not reset, use loc to make idx unique across folds
        meta = self.dataframes[self.dataset].iloc[idx]

        emb1 = self.embeddings[meta['p1']]
        emb2 = self.embeddings[meta['p2']]

        return emb1, emb2, meta['label'], meta.to_dict()


class BFWImages(Dataset):
    def __init__(self,
                 data_root: str = './data/bfw/uncropped-face-samples/',
                 csv_file: str = './data/bfw/bfw-v0.1.5-datatable.csv',
                 transform: Optional[Compose] = None,
                ):
        """
        TODO

        Parameters:
            data_root: str - Path from which data can be found
            csv_file: str - CSV file containing all information about the data (paths to images, labels, etc)
            transform: Optional[Compose] - Optionally add some transformations when reading data
        """

        # Convert to global paths for interpretable printing
        data_root = os.path.abspath(data_root)
        csv_file = os.path.abspath(csv_file)

        # Save location of data
        self.data_root = data_root

        # Open CSV as 'data', reading of images happesn in BFWFold
        df = pd.read_csv( csv_file )

        names_left = {
            'p1': 'path',
            'id1': 'id',
            'att1': 'attribute long',
            'a1': 'attribute',
            'g1': 'gender',
            'e1': 'ethnicity'
        }
        group_left = df[ list(names_left.keys()) ].rename(columns=names_left).groupby('path').first()

        names_right = {
            'p2': 'path',
            'id2': 'id',
            'att2': 'attribute-long',
            'a2': 'attribute',
            'g2': 'gender',
            'e2': 'ethnicity'
        }
        group_right = df[ list(names_right.keys()) ].rename(columns=names_right).groupby('path').first()

        unique_images = pd.concat([group_left, group_right])
        unique_images = unique_images.reset_index().drop_duplicates(subset='path')

        self.dataframe = unique_images

        self.transform = transform


    def __len__(self):
        """Give length of DataSet"""
        return len(self.dataframe)


    def __getitem__(self, idx):
        """
        Given some index, give images and label corresponding to that pair

        Parameters:
            idx: ? - Index of the sample TODO: what type is idx?

        Returns:
            WIP
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Use iloc if index is not reset, use loc to make idx unique across folds
        row = self.dataframe.iloc[idx]

        # img = io.imread( os.path.join(self.data_root, row['path']) )
        img = Image.open( os.path.join(self.data_root, row['path']) )

        if self.transform:
            img = self.transform(img)

        return img, row.to_dict()