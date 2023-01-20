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


class RFWImages(Dataset):
    def __init__(self,
                 data_root: str = './',
                 csv_file: str = './data/rfw/filtered_pairs.csv',
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
            'filepath1': 'path',
            'id1': 'id',
        }
        group_left = df[ list(names_left.keys()) ].rename(columns=names_left).groupby('path').first()

        names_right = {
            'filepath2': 'path',
            'id2': 'id',
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