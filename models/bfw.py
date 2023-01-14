import os
from typing import Optional

import pandas as pd

from skimage import io

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

class BFWFold(Dataset):
    def __init__(self, 
                 dataframe: pd.DataFrame,
                 folds: int|list,
                 data_root: str = './data/bfw/',
                 transform: Optional[callable] = None
                ):
        """
        Initialize a subset of the BFW dataset for the provided folds

        Provided dataframe contains columns 'fold','p1','p2' and 'label', 
        providing what fold each pair belongs to, the paths to two images
        and a label whether they are the same person

        Method will save the part of the dataframe of the provided folds, data root and
        transformation.

        Parameters:
            dataframe: pd.DataFrame - Dataframe having one image pair per row
            folds: int|list - One or multiple folds to take from the dataframe
            data_root: str - path to folder containing images
            transform: Optional[any] - A transformation applied to any image being read
        """

        # Save this for when reading the data
        self.data_root = data_root

        # For consistency, convert int to list of int
        if isinstance(folds, int):
            folds = [folds,]

        # Filter dataframe for provided folds
        self.dataframe = dataframe[ dataframe['fold'].isin(folds) ]

        # Uncomment to reset index such that it ranges 0,N. It now will contain holes in the index!
        # Remember to change `BFWFold.dataframe.iloc` to `BFWFold.dataframe.loc` 
        # in `BFWFold.__getitem__` if you do this, otherwise it has no effect on the data returned
        
        # self.pairs_df.reset_index(inplace=True, names='global_index')

        # Save transforms
        self.transform = transform

    def __len__(self):
        """Give length of DataSet"""
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Given some index, give images and label corresponding to that pair
        TODO: Also return persion ID and sensitive attributes?

        Parameters:
            idx: ? - Index of the sample TODO: what type is idx?
        
        Returns:
            img1: np.ndarray - First image of sample pair
            img2: np.ndarray - Second image of sample pair
            label: int - 1 if image pair is of the same person, 0 otherwise
        """
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
                 augmentation: Optional[Compose] = None
                ):
        """
        Set up BFW dataset as a combination of folds.
        Method will initialize by reading the given csv file
        Then use `trainset, testset = bfw.fold(k)` get the kth fold specified in the csv file

        It will save given transform as basic processing of the data, such as .toTensor
        If provided, augmentations will additionally applied to ONLY the train set

        Parameters:
            data_root: str - Path from which data can be found
            csv_file: str - CSV file containing all information about the data (paths to images, labels, etc)
            transform: Optional[Compose] - Optionally add some transformations when reading data, will be aplied to train and test data
            augmentation: Optional[Compose] - Optionally add some agumentations, only applied to train set
        """
        
        # Convert to global paths for interpretable printing
        data_root = os.path.abspath(data_root)
        csv_file = os.path.abspath(csv_file)

        # Save location of data
        self.data_root = data_root
        
        # Open CSV as 'data', reading of images happesn in BFWFold
        self.dataframe = pd.read_csv( csv_file )

        # Save which folds are available
        self.folds = self.dataframe['fold'].unique()

        # Train transformations also include augmentations
        self.train_transform = Compose([transform, augmentation])

        # Test transformations
        self.test_transform = transform

    def fold(self, k: int) -> tuple[BFWFold, BFWFold]:
        """
        Given k, return the train and test set of the kth fold of the BFW dataset
        It initializes these folds using the dataframe it loaded in init

        Parameters:
            k: int - What fold to return
        
        Returns:
            trainSet: BFWFold - The train set, consisting of data where fold != k
            testSet: BFWFold - The test set, consiting of data where fold == k
        """
        # Make sure provided fold exists
        if k not in self.folds:
            raise KeyError(f"Fold {k} not found in BFW dataset")
        
        # Trainfolds consist of all other folds
        trainFolds = [fold for fold in self.folds if fold != k]
        trainSet = BFWFold(dataframe=self.dataframe,
                           folds=trainFolds,
                           data_root=self.data_root,
                           transform=self.train_transform)

        # Testset from given fold
        testSet = BFWFold(dataframe=self.dataframe,
                          folds=[k,],
                          data_root=self.data_root,
                          transform=self.test_transform)
        
        # TODO: Convert to dataloaders?

        return trainSet, testSet