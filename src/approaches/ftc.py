import pickle

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import roc_curve
from torch import nn
from torch.utils.data import DataLoader

from argparse import Namespace

from calibrationMethods import BetaCalibration
from dataset import Dataset

def ftc(dataset: Dataset, conf: Namespace) -> np.ndarray:
    """TODO"""
    # Set up dataframe from dataset
    df = dataset.df.copy()
    df['test'] = (df['fold'] == dataset.fold)
    df['score'] = df[dataset.feature]

    loader_train = get_loader(dataset=dataset, train=True)
    loader_test = get_loader(dataset=dataset, train=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FTCNN(device=device)
    loss_fn = fair_individual_loss # TODO

    optimizer = optim.Adam(
        params=model.parameters(),
        lr=1e03,
        weight_decay=1e-3,
    )

    epochs=50
    for _ in range(epochs):
        train_loop(
            model=model,
            dataloader=loader_train,
            device=device,
            loss_fn=loss_fn,
            optimizer=optimizer
        )
        test_loop(
            dataloader=loader_train,
            model=model,
            device=device,
            loss_fn=loss_fn
        )

    test_loop(dataloader=loader_test, model=model, loss_fn=loss_fn)

    model.cpu()

    return

def fair_individual_loss(CELoss: nn.CrossEntropyLoss, subgroups: list[str]):
    def _fair_individual_loss(g1: str, g2: str, y: torch.Tensor[bool], logits: torch.Tensor[float]):
        """
        Parameters:
            g1: str - group left image, some items in `subgroups`
            g2: str - group right image, some items in `subgroups`
            y: torch.Tensor[bool] - A 1D array of size batchsize with bools, indicating the true class
            logits: torch.Tensor[float] - A Bx2 array of logits, where B is batch size

        Returns:
            loss: float - The fair individual loss
        """
        loss = 0.

        # Try each combination of subgroups
        for sub1 in subgroups:
            select1 = (g1 == sub1) & (g2 == sub1)
            n1 = np.count_nonzero(select1)
            for sub2 in subgroups:
                select2 = (g1 == sub2) & (g2 == sub2)
                n2 = np.count_nonzero(select2)

                # If some data exists in this combination of subgroups...
                if n1 > 0 and n2 > 0:
                    # Compute Fair Individual Loss
                    select = y[select1] == y[select2]
                    aux = torch.cdist(logits[select1, :], logits[select2, :])[select].pow(2).sum()
                    loss += aux/(n1*n2)

        return .5 * CELoss(logits, y) + .5*loss

    return _fair_individual_loss


def get_loader(dataset: Dataset, train:bool=False) -> DataLoader:
    embeddings, idx2path = dataset.get_embeddings(train=train, return_mapper=True)
    path2idx = dict((path,idx) for idx,path in enumerate(idx2path))

    df = dataset.df.copy()
    if train:
        df = df[ df['fold'] != dataset.fold ]
    else:
        df = df[ df['fold'] == dataset.fold ]

    labels = df['same'].to_numpy(dtype=bool)

    attributes = list(dataset.consts['sensitive_attributes'].keys())
    cols_left = [ dataset.consts['sensitive_attributes'][key]['cols'][0] for key in attributes ]
    cols_right = [ dataset.consts['sensitive_attributes'][key]['cols'][1] for key in attributes ]

    idxs_left = []
    idxs_right = []
    attr_left = []
    attr_right = []

    for _,row in df.iterrows():
        idxs_left.append( path2idx[row['path1']] )
        idxs_right.append( path2idx[row['path2']] )
        attr_left.append( "_".join( str(row[key]) for key in cols_left ) )
        attr_right.append( "_".join( str(row[key]) for key in cols_right ) )

    data = EmbeddingsDataset(
        embeddings=embeddings,
        idxs_left=idxs_left,
        idxs_right=idxs_right,
        attr_left=attr_left,
        attr_right=attr_right,
        labels=labels
    )

    return DataLoader(
        dataset=data,
        batch_size=200, # TODO
        shuffle=False,
        num_workers=0
    )


class EmbeddingsDataset(torch.utils.data.Dataset):
    """TODO"""

    def __init__(
        self,
        embeddings: np.ndarray[float],
        idxs_left: list[int],
        idxs_right: list[int],
        attr_left: list[str],
        attr_right: list[str],
        labels: np.ndarray[bool],
    ) -> None:
        """
        TODO
        """
        self.embeddings = torch.from_numpy(embeddings)
        self.idxs_left = idxs_left
        self.attr_left = attr_left
        self.idx_right = idxs_right
        self.attr_right = attr_right
        self.labels = torch.from_numpy(labels).type(torch.LongTensor)

        self.len = len(idxs_left)

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, str, bool]:
        emb_diff = torch.abs(self.embeddings[self.idxs_left[idx]] - self.embeddings[self.idx_right[idx]])
        return emb_diff, self.attr_left[idx], self.attr_right[idx], self.labels[idx]


class FTCNN(nn.Module):
    """TODO"""
    def __init__(self, device: torch.device):
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

        self.model.to(device)

    def forward(self, x):
        return self.model(x)

def train_loop(model, dataloader, device, loss_fn, optimizer):
    """TODO"""
    model.train()
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

@torch.no_grad()
def test_loop(dataloader, model, device, loss_fn):
    """TODO"""
    size = len(dataloader.dataset)
    test_loss, correct = 0., 0.

    scores = torch.zeros(0, 2)
    ground_truth = torch.zeros(0)

    model.eval()
    model.to(device)
    for X,_,_,y in dataloader:
        X: torch.Tensor = X.to(device)
        y: torch.Tensor = y.to(device)

        logits: torch.Tensor = model(X)

        test_loss += loss_fn(logits, y).item()

        correct += (logits.argmax(dim=1) == y).type(torch.float).sum().item()

        scores = torch.cat((scores, logits.softmax(dim=1)))

        ground_truth = torch.cat([ground_truth, y], 0)

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    fpr, tpr, thr = roc_curve(ground_truth, scores[:, 1].numpy())
    print('FNR @ 0.1 FPR %1.2f'% (1-tpr[np.argmin(np.abs(fpr-1e-3))]))
    return scores, ground_truth
