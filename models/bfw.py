import os
from typing import Union,Optional

import numpy as np
import pandas as pd

from skimage import io
from sklearn.model_selection import KFold

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.transforms import Compose

class BFWFold(Dataset):
    def __init__(self, 
    dataframe: pd.DataFrame,
    folds: Union[int,list],
    data_root: str = './data/bfw/',
    transform: Optional[Compose] = None):

        self.data_root = data_root

        if isinstance(folds, int):
            folds = [folds,]

        self.dataframe = dataframe[ dataframe['fold'].isin(folds) ]

        #Uncomment to reset index such that it ranges 0,N. It now will contain holes!
        # self.pairs_df.reset_index(inplace=True, names='global_index')

        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Use iloc if index is not reset, use loc to make idx unique across folds
        row = self.dataframe.iloc[idx]

        img1 = io.imread( os.path.join(self.data_root, row['p1']) )
        img2 = io.imread( os.path.join(self.data_root, row['p2']) )
        label = int(row['label'])

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return (img1, img2), label
        # return (img1, img2), label, row['global_index']


class BFW():
    def __init__(self,
    data_root: str = './data/bfw',
    csv_file: str = './data/bfw/bfw-v0.1.5-datatable.csv',
    transform: Optional[Compose] = None,
    augmentation: Optional[Compose] = None):
        data_root = os.path.abspath(data_root)
        csv_file = os.path.abspath(csv_file)

        self.data_root = data_root
        self.csv_file = csv_file
        
        self.dataframe = pd.read_csv( csv_file )
        self.folds = self.dataframe['fold'].unique()
        self.train_transform = Compose([transform, augmentation])
        self.test_transform = transform

    def fold(self, k: int):
        if k not in self.folds:
            raise KeyError(f"Fold {k} not found in BFW dataset")
        
        trainFolds = [fold for fold in self.folds if fold != k]
        trainSet = BFWFold(dataframe=self.dataframe,
                           folds=trainFolds,
                           data_root=self.data_root,
                           transform=self.train_transform)

        testSet = BFWFold(dataframe=self.dataframe,
                          folds=[k,],
                          data_root=self.data_root,
                          transform=self.test_transform)
        
        # TODO: Convert to dataloaders?

        return trainSet, testSet