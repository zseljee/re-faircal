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

    if not os.path.isfile(savepath):
        print("Training...")
        loader_train = get_loader(dataset=dataset, train=True)
        loader_test = get_loader(dataset=dataset, train=False)

        # TODO very hacky, could be cleaner
        subgroups = set()
        for _,g1,g2,_ in loader_train:
            subgroups |= set(g1) | set(g2)
        for _,g1,g2,_ in loader_test:
            subgroups |= set(g1) | set(g2)
        subgroups = list(subgroups)

        model, test_loss, acc, fnr = train(
            model=model,
            device=device,
            loaders={'train': loader_train, 'test': loader_test},
            epochs=5,
        )
        print(f"Test: loss={test_loss:.2e}, acc={acc:4.1f}%, fnr={fnr:.1f}")
        print("\nSaving model to", savepath)
        torch.save(model.state_dict(), savepath)
    else:
        print("Loading model from", savepath)
        model.load_state_dict(torch.load(savepath))

    loader = get_loader(dataset=dataset, train=None, shuffle=False)

    scores, _,_,_ = test_loop(dataloader=loader, model=model, device=device, loss_fn=nn.CrossEntropyLoss())


    return scores[:,1].detach().cpu().numpy()


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
    subgroups = ['Asian', 'African', 'Caucasian', 'Indian']
    loss = 0.
    for i in subgroups:
        for j in subgroups:
            select_i = np.logical_and(np.array(g1) == i, np.array(g2) == i)
            select_j = np.logical_and(np.array(g1) == j, np.array(g2) == j)
            if (sum(select_i) > 0) and (sum(select_j) > 0):
                select = y[select_i].reshape(-1, 1) == y[select_j]
                aux = torch.cdist(logits[select_i, :], logits[select_j, :])[select].pow(2).sum()
                loss += aux/(sum(select_i)*sum(select_j))
    return loss


def get_loader(dataset: Dataset,
               train: bool|None=False,
               shuffle: bool = True,
               batch_size: int = 200,
              ) -> DataLoader:
    embeddings, idx2path = dataset.get_embeddings(train=train, return_mapper=True)
    path2idx = dict((path,idx) for idx,path in enumerate(idx2path))

    df = dataset.df.copy()
    if train == True:
        df = df[ df['fold'] != dataset.fold ]
    elif train == False:
        df = df[ df['fold'] == dataset.fold ]
    df.reset_index(inplace=True)

    labels = df['same'].to_numpy(dtype=bool)

    attributes = list(dataset.consts['sensitive_attributes'].keys())
    cols_left = [ dataset.consts['sensitive_attributes'][key]['cols'][0] for key in attributes ]
    cols_right = [ dataset.consts['sensitive_attributes'][key]['cols'][1] for key in attributes ]

    attr_left = []
    attr_right = []
    error_embeddings = torch.zeros((len(df), 512))

    for i,row in df.iterrows():
        error_embeddings[i,:] = torch.from_numpy(embeddings[path2idx[row['path1']]] - embeddings[path2idx[row['path2']]])
        attr_left.append( "_".join( str(row[key]) for key in cols_left ) )
        attr_right.append( "_".join( str(row[key]) for key in cols_right ) )

    error_embeddings = torch.abs(error_embeddings)

    data = EmbeddingsDataset(
        error_embeddings=error_embeddings,
        attr_left=attr_left,
        attr_right=attr_right,
        labels=labels
    )

    return DataLoader(
        dataset=data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )



class EmbeddingsDataset(torch.utils.data.Dataset):
    """TODO"""

    def __init__(
        self,
        error_embeddings: torch.Tensor,
        attr_left: list[str],
        attr_right: list[str],
        labels: np.ndarray[bool],
    ) -> None:
        """
        TODO
        """
        self.error_embeddings = error_embeddings

        self.attr_left = attr_left
        self.attr_right = attr_right

        self.labels = torch.zeros(len(labels)).type(torch.LongTensor)
        self.labels[labels] = 1


    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, str, bool]:
        return self.error_embeddings[idx], self.attr_left[idx], self.attr_right[idx], self.labels[idx]


class FTCNN(nn.Module):
    """TODO"""
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

    assert 'train' in loaders and 'test' in loaders, "Please specify a train and test set in `loaders` parameter."

    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        params=model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    pbar = None
    test_loss, acc, fnr = 0., 0., 0.
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
        _, test_loss, acc, fnr = test_loop(
            dataloader=loaders['val' if 'val' in loaders else 'test'],
            model=model,
            device=device,
            loss_fn=loss_fn
        )


        if pbar is not None:
            pbar.set_postfix(loss=f"{test_loss:.2e}", acc=f"{acc:4.1f}%", fnr=f"{fnr:.1f}", refresh=False)
            pbar.update(1)
        else:
            print(f"Epoch {epoch:4}: loss={test_loss:.2e}, acc={acc:4.1f}%, fnr={fnr:.1f}")

    if pbar is not None:
        pbar.close()
        print('\n')

    _, test_loss, acc, fnr = test_loop(
        dataloader=loaders['test'],
        model=model,
        device=device,
        loss_fn=loss_fn
    )

    model.cpu()

    return model, test_loss, acc, fnr


def train_loop(epoch, model, dataloader: DataLoader, device, loss_fn, optimizer, pbar=None):
    """TODO"""
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
        loss = loss_fn(logits, y) *.5 + .5*fair_individual_loss(g1, g2, y, logits)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()


@torch.no_grad()
def test_loop(dataloader: DataLoader, model: FTCNN, device: torch.device, loss_fn: callable):
    """TODO"""
    size = len(dataloader.dataset)
    test_loss, correct = 0., 0.

    scores = torch.zeros(0, 2)
    ground_truth = torch.zeros(0)

    model.eval()
    model.to(device)

    pbar = None
    if DO_PBAR:
        pbar = tqdm.tqdm(dataloader, desc="Validating", position=1, leave=False, unit="pair", ncols=150, unit_scale=dataloader.batch_size)

    for X,g1,g2,y in dataloader:
        X: torch.Tensor = X.to(device)

        logits: torch.Tensor = model(X).cpu()

        test_loss += loss_fn(logits, y).item()

        correct += torch.count_nonzero(logits.argmax(dim=1) == y)

        scores = torch.cat((scores, logits.softmax(dim=1)))

        ground_truth = torch.cat([ground_truth, y], 0)

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    test_loss /= size
    acc = (correct / size) * 100.
    fpr, tpr, thr = roc_curve(ground_truth, scores[:, 1].numpy())
    fnr = 1. - np.interp(1e-3, fpr, tpr)

    return scores, test_loss, acc, fnr


def get_savepath(dataset: Dataset, conf:Namespace) -> str:

    dirname = os.path.join(
        EXPERIMENT_FOLDER,
        "FTCmodels",
    )
    os.makedirs(dirname, exist_ok=True)

    fname = os.path.join(
        dirname,
        f"FTCNN_{dataset.name}_{conf.feature}_fold{dataset.fold}.pt"
    )
    return fname
