import pickle

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import roc_curve
from torch import nn
from torch.utils.data import DataLoader
import tqdm
import os

from argparse import Namespace

from calibrationMethods import BetaCalibration
from dataset import Dataset
from constants import EXPERIMENT_FOLDER

DO_PBAR = True

def ftc(dataset: Dataset, conf: Namespace) -> np.ndarray:
    """TODO"""
    print("Setting up...")
    # Set up dataframe from dataset
    df = dataset.df.copy()
    df['test'] = (df['fold'] == dataset.fold)
    df['score'] = df[dataset.feature]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device", device)

    model = FTCNN()

    savepath = get_savepath(dataset=dataset, conf=conf)

    loader_train = get_loader(dataset=dataset, train=True, shuffle=False)
    loader_test = get_loader(dataset=dataset, train=False, shuffle=False)

    if not os.path.isfile(savepath):
        print("Training...")

        model, val_losses, test_loss, acc, fnr = train(
            model=model,
            device=device,
            loaders={'train': loader_train, 'test': loader_test},
        )
        print(f"Test: loss={test_loss:.2e}, acc={acc:4.1f}%, fnr={fnr:.1f}")
        print("\nSaving model to", savepath)
        torch.save(model.state_dict(), savepath)
    else:
        print("Loading model from", savepath)
        model.load_state_dict(torch.load(savepath))

    scores, ground_truth, _,_,_ = test_loop(dataloader=loader_train, model=model, device=device, loss_fn=None)

    calibrator = BetaCalibration(
        scores=scores[:,1].numpy(),
        ground_truth=ground_truth.numpy(),
        score_max=0.,
        score_min=1.,
    )


    return calibrator.predict(df['score'])


def get_loss_fn(subgroups: list[str], lamb:float = .5):
    CELoss = nn.CrossEntropyLoss()

    def fair_individual_loss(g1: list[str], g2: list[str], y: torch.Tensor, logits: torch.Tensor):
        """
        Parameters:
            g1: list[str] - group left image, batched. Contains items from `subgroups`
            g2: list[str] - group right image, batched. Contains items from `subgroups`
            y: torch.Tensor[bool] - A 1D array of size batchsize with bools, indicating the true class
            logits: torch.Tensor[float] - A Bx2 array of logits, where B is batch size

        Returns:
            loss: float - The fair individual loss
        """
        g1 = np.array(g1)
        g2 = np.array(g2)
        loss = 0.
        for i in subgroups:
            select_i = np.logical_and(g1 == i, g2 == i)
            for j in subgroups:
                select_j = np.logical_and(g1 == j, g2 == j)
                if (sum(select_i) > 0) and (sum(select_j) > 0):
                    select = y[select_i].reshape(-1, 1) == y[select_j]
                    aux = torch.cdist(logits[select_i, :], logits[select_j, :])[select].pow(2).sum()
                    loss += aux/(sum(select_i)*sum(select_j))

        return lamb*loss + (1.-lamb)*CELoss(logits, y)

    return fair_individual_loss


def get_loader(dataset: Dataset,
               train: bool|None=False,
               shuffle: bool = True,
               batch_size: int = 200,
              ) -> DataLoader:
    """
    Set up a DataLoader from a Dataset instance using the EmbeddingsDataset class

    From the dataset, obtain the necessary embeddings, convert them to
    'error embeddings' and obtain the attribute and label for each image pair.

    Error embeddings are computed using abs( v - w ), where v is the embedding
    of the left image, w is the embedding of the right image and abs is applied
    element-wise.

    Parameters:
        dataset: Dataset - A dataset instance (note this is a class in this project,
        not a torch.utils.data.Dataset)
        train: bool|None - If True, only use train data, if False use all
        except train data, if None, use all data
        shuffle: bool - Whether to make a shuffled DataLoader
        batch_size: int - What batch size to use

    Returns:
        loader: DataLoader - A torch.utils.data.DataLoader instance
    """

    # Get embeddings + mappers from dataset
    embeddings, idx2path = dataset.get_embeddings(train=train, return_mapper=True)
    path2idx = dict((path,idx) for idx,path in enumerate(idx2path))

    # Get dataframe
    df = dataset.df.copy()
    if train == True:
        df = df[ df['fold'] != dataset.fold ]
    elif train == False:
        df = df[ df['fold'] == dataset.fold ]

    # Reset index to get a range(0, N) (for indexing error_embeddings)
    df.reset_index(inplace=True)

    # Use np.ndarray for labels
    labels = df['same'].to_numpy(dtype=bool)

    # Get the attributes that exist in this dataset
    attributes = list(dataset.consts['sensitive_attributes'].keys())
    # The columns where those attributes can be found for the left and right image
    cols_left = [ dataset.consts['sensitive_attributes'][key]['cols'][0] for key in attributes ]
    cols_right = [ dataset.consts['sensitive_attributes'][key]['cols'][1] for key in attributes ]

    # Save a string indicating the subgroup of each individual
    attr_left = []
    attr_right = []
    # Save 'error embeddings'
    error_embeddings = torch.zeros((len(df), 512))

    for i,row in df.iterrows():
        #convert embedding pair to error embedding
        error_embeddings[i,:] = torch.from_numpy(embeddings[path2idx[row['path1']]] - embeddings[path2idx[row['path2']]])

        # Subgroup is defined as combination of attributes, concatenate those to get unique string for each subgroup
        # ie ethnicity='African', gender='Male' gives 'African_Male'
        attr_left.append( "_".join( str(row[key]) for key in cols_left ) )
        attr_right.append( "_".join( str(row[key]) for key in cols_right ) )

    # Take absolute values
    error_embeddings = torch.abs(error_embeddings)

    # Set up torch.utils.data.Dataset instance
    data = EmbeddingsDataset(
        error_embeddings=error_embeddings,
        attr_left=attr_left,
        attr_right=attr_right,
        labels=labels
    )

    # Set up DataLoader
    return DataLoader(
        dataset=data,
        batch_size=batch_size,
        shuffle=shuffle,
    )


class EmbeddingsDataset(torch.utils.data.Dataset):
    """
    Custom dataset to work with FTC approach
    Given a list of error embeddings, attributes and labels
    (all of equal length), give a dataset that yields the ith element
    of each of those lists for __getitem__.
    """

    def __init__(
        self,
        error_embeddings: torch.Tensor,
        attr_left: list[str],
        attr_right: list[str],
        labels: np.ndarray[bool],
    ) -> None:
        """
        let N be the number of image pairs in the dataset, train the network
        using abs( v - w ), where v is the embedding of the left image and w is
        the embedding of the right image. Take the absolute of these values
        as 'error embedding'.

        During training, use the fair_individual_loss based on the subgroups,
        so also save the subgroups.

        Parameters:
            error_embeddings: torch.Tensor - A Nx512 Tensor with embeddings
            attr_left, attr_right: list[str] - A list of size N containing strings
            as unique identifiers.
            labels: np.ndarray - A ndarray of shape (N,) containing bools indicating
            genuine/imposter pairs.
        """
        self.error_embeddings = error_embeddings

        self.attr_left = attr_left
        self.attr_right = attr_right

        # Convert mask to {0,1} as longTensor
        self.labels = torch.zeros(len(labels)).type(torch.LongTensor)
        self.labels[labels] = 1

        self.subgroups = set(self.attr_left) | set(self.attr_right)


    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, str, bool]:
        return self.error_embeddings[idx], self.attr_left[idx], self.attr_right[idx], self.labels[idx]


class FTCNN(nn.Module):
    """
    FTC Neural Network, adapted from Terhörst et al. (2020) by rescaling
    the layer sizes to work with embedding size 512.
    """
    def __init__(self):
        super(FTCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(128*4, 256*4),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256*4, 512*4),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512*4, 512*4),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512*4, 2),
        )

    def forward(self, x):
        return self.model(x)


def train(model: FTCNN,
          device: torch.device,
          loaders: dict[str, DataLoader],
          epochs: int = 50,
          lr: float = 1e-3,
          weight_decay: float = 1e-3
    ) -> FTCNN:
    """
    General train function. It sets up optimizer and loss function, then
    repeatedly calls `train_loop` on train set, `test_loop` on val set
    and ends with one `test_loop` on the test set.

    Parameters:
        model: FTCNN - Model to train
        device: torch.device - device to train with
        loaders: dict[str, DataLoader] - A dictionary with keys 'train', 'test'
        and (optionally) 'val' and DataLoaders as values
        epochs: int - The number of epochs to run
        lr: float - Learning rate to use
        weight_decay: float - Weight decay in Adam optimizer

    Returns:
        model: FTCNN - Trained model
        val_losses: list[float] - Validation loss per iteration (zero-indexed, compted after training an epoch)
        test_loss: float - The final test loss
        acc: float - Accuracy on the test set
        fnr: float - FNR @ .1% FPR on the test set
    """

    assert 'train' in loaders and 'test' in loaders, "Please specify a train and test set in `loaders` parameter."

    subgroups = list(set(loaders['train'].dataset.subgroups) | set(loaders['test'].dataset.subgroups))
    loss_fn = get_loss_fn(subgroups=subgroups, lamb=.5)

    optimizer = optim.Adam(
        params=model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    pbar = None
    test_loss, acc, fnr = 0., 0., 0.
    val_losses = []
    if DO_PBAR:
        pbar = tqdm.tqdm(desc="Training  ", total=epochs, mininterval=.5, unit="epoch", ncols=150, position=0, leave=False)
        pbar.set_postfix(loss=f"{test_loss:.2e}", acc=f"{acc:4.1f}%", fnr=f"{fnr:.1f}")

    for epoch in range(epochs):
        train_loop(
            epoch=epoch,
            model=model,
            dataloader=loaders['train'],
            device=device,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
        _, _, val_loss, acc, fnr = test_loop(
            dataloader=loaders['val' if 'val' in loaders else 'test'],
            model=model,
            device=device,
            loss_fn=loss_fn
        )
        val_losses.append(val_loss)


        if pbar is not None:
            pbar.set_postfix(loss=f"{val_loss:.2e}", acc=f"{acc:4.1f}%", fnr=f"{fnr:.1f}", refresh=False)
            pbar.update(1)
        else:
            print(f"Epoch {epoch:4}: loss={test_loss:.2e}, acc={acc:4.1f}%, fnr={fnr:.1f}")

    if pbar is not None:
        pbar.close()
        print() # To clear the TQDM semi-empty line

    _, _, test_loss, acc, fnr = test_loop(
        dataloader=loaders['test'],
        model=model,
        device=device,
        loss_fn=loss_fn
    )
    print() # To clear the TQDM semi-empty line

    model.cpu()

    return model, val_losses, test_loss, acc, fnr


def train_loop(epoch: int, model: FTCNN, dataloader: DataLoader, device: torch.device, loss_fn: callable, optimizer: optim.Optimizer) -> None:
    """
    Standard train loop, ie one epoch. Loads data from the dataloader, executes
    a forward pass through the model and a backward pass using the optimizer.

    Parameters:
        epoch: int - the current epoch
        model: FTCNN - Model to train on
        dataloader: DataLoader - Dataloader for data
        device: torch.device - Device to use
        loss_fn: callable - Some loss function to run; the total loss if .5*loss_fn
        + .5*fair individual loss (which is fixed)
        optimizer: torch.optim.Optimizer - Some optimizer
    """
    model.train()
    model.to(device)

    pbar = None
    if DO_PBAR:
        pbar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch:4}", position=1, leave=False, unit="pair", ncols=150, unit_scale=dataloader.batch_size)

    for X, g1, g2, y in dataloader:
        X = X.to(device)
        y = y.to(device)
        # Compute prediction and loss
        logits = model(X)
        loss = loss_fn(g1, g2, y, logits)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()


@torch.no_grad()
def test_loop(dataloader: DataLoader, model: FTCNN, device: torch.device, loss_fn: callable) -> tuple[torch.Tensor, float, float, float]:
    """
    Standard test loop for Torch. Iterate over given dataloader, keep track of
    the running #correct, output and ground truth and return some metrics.

    Parameters:
        dataloader: DataLoader - A torch.utils.data.DataLoader instance to load data from
        model: FTCNN - Model to evaluate
        device: torch.device - Device to run experiment on
        loss_fn: callable - Some function to call with signature
        `loss: torch.Tensor = loss_fn(logits: torch.Tensor, y: torch.Tensor)`

    Returns:
        scores: torch.Tensor - A Nx2 tensor containing the output for each item in DataLoader
        ground_truth: torch.Tensor - A Nx2 tensor containing the target for each item in DataLoader
        test_loss: float - Loss on test set
        acc: float - Accuracy, in range [0, 100]
        fnr: float - FNR @ .1% FPR, computed from scores and ground truth
    """
    # Use dataset size to normalize scores
    size = len(dataloader.dataset)

    test_loss, correct = 0., 0.
    scores = torch.zeros(0, 2)
    ground_truth = torch.zeros(0)

    # Set model to appropriate settings
    model.eval()
    model.to(device)

    # Set up progress bar if specified to globally
    pbar = None
    if DO_PBAR:
        pbar = tqdm.tqdm(dataloader, desc="Validating", position=1, leave=False, unit="pair", ncols=150, unit_scale=dataloader.batch_size)

    # Iterate data
    for X,g1,g2,y in dataloader:
        # Send data to device
        X: torch.Tensor = X.to(device)

        # Forward pass, send back to CPU
        logits: torch.Tensor = model(X).cpu()

        # Add metrics
        if not isinstance(loss_fn, nn.CrossEntropyLoss) and loss_fn is not None:
            test_loss += loss_fn(g1, g2, y, logits).item()

        correct += torch.count_nonzero(logits.argmax(dim=1) == y)

        scores = torch.cat((scores, logits.softmax(dim=1)))

        ground_truth = torch.cat([ground_truth, y], 0)

        # Update progress bar
        if pbar is not None:
            pbar.update(1)

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Finish up metrics
    test_loss /= size
    acc = (correct / size) * 100.

    fpr, tpr, thr = roc_curve(ground_truth, scores[:, 1].numpy())

    # Use np.interp to set up TPR as a function of FPR, evaluate at FPR=1e-3
    # Then, FNR = 1 - TPR
    fnr = 1. - np.interp(1e-3, fpr, tpr)

    return scores, ground_truth, test_loss, acc, fnr


def get_savepath(dataset: Dataset, conf:Namespace) -> str:
    """
    Using the information in the dataset and current configuration,
    give the path to a (possibly non-existing) model. Can be used to
    have a consistent file structure and easy method to load models
    on a specific configuration.

    Returned path will always be in `EXPERIMENTS_FOLDER` defined in
    `constants.py`; will create any intermediate directories if necessary

    Learning rate, weight decay and #epochs are considered constant and are
    therefore not taken into account.

    Parameters:
        dataset: Dataset - A Dataset instance (as defined in this project, not
        a torch.utils.data.Dataset instance!)
        conf: Namespace - The configuration of this model/approach

    Returns:
        path: str - A complete path to save a model to / load a model from
    """

    # First set up the folder
    dirname = os.path.join(
        EXPERIMENT_FOLDER,
        "FTCmodels",
    )

    # Make sure parent folder exists
    os.makedirs(dirname, exist_ok=True)

    # Then add file name
    path = os.path.join(
        dirname,
        f"FTCNN_{dataset.name}_{conf.feature}_fold{dataset.fold}.pt"
    )
    return path
