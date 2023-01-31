import numpy as np
import os
import pandas as pd
import pickle
import torch
import tqdm

from argparse import Namespace
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ArcFace is usually not available
# TODO: uncomment if you use ArcFace
# from arcface import ArcFace
from constants import *
from gen_rfw_table import generate_rfw_df


device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
cpu = torch.device('cpu')
print("Using device:", device)

# If cropping has already been done, skip it by setting this to True
SKIP_CROP = False

# Set resize for MTCNN, if not resizing, set to None
RESIZE = (400,400)

# Batch size, improves performance. Does not work without resizing! See `crop()`
BATCH_SIZE = 64

# Whether to display a TQDM progress bar
DO_PBAR = True


class PathToImage(Dataset):
    """
    PathToImage class, for using dataloaders and batches on a list of image
    paths.

    Initialize with a root and a list of paths, use with a dataloader and
    collate function to give batches of PIL Images (as a list[PIL]).

    Collate function is defined below!

    Optionally, give a tuple (i,j) to resize images, in case your model only
    accepts same-sized images.

    Optionaly, give a torchvision.transforms.Compose to transform the image
    before returning.

    Usage:
    >>> image_root = '/home/myName/myProject/data/' # First common folder of data
    >>> paths = ['folder1/a.jpg', 'folder2/b.jpg', ...] # Path from `image_root`
    >>> resize = (400, 200) # Make sure all images are of same size (optional)
    >>> dataset = PathToImage(image_root, paths, resize)
    >>> loader = torch.utils.data.DataLoader(dataset, batch_size=64, collate_fn=collate)
    >>> for paths, images in loader:
    >>>     paths: list[str]
    >>>     images: list[PIL.Image.Image]
    """

    def __init__(self,
                 image_root: str,
                 paths: list[str],
                 resize: None|tuple[int, int]=None,
                 transform: None|transforms.Compose=None):
        """
        Parameters:
            image_root: str - First common directory of all images
            paths: list[str] - List of paths, starting from image_root
            resize: tuple[int,int]|None - Optionally, resize images to this size using `'PIL Image'.resize`
        """
        self.image_root = image_root
        self.paths = paths
        self.len = len(paths)
        self.resize = resize
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int) -> tuple[str, Image.Image]:
        # Open image whose path is at index idx
        I = Image.open( os.path.join( self.image_root, self.paths[idx]) )

        # Resize if specified
        if self.resize is not None:
            I = I.resize(size=self.resize)

        # Apply transform
        if self.transform is not None:
            I = self.transform(I)

        # Return tuple path, image
        return [self.paths[idx], I]


def collate(batch: list[tuple[str, Image.Image]]) -> tuple[list[str], list[Image.Image]]:
    """
    Collate function to use with PathToImage class

    The torch dataloader will append the tuples returned by PathToImage.__getitem__
    which results in a batch list[tuple[str, Image]],
    while it is prefered to have tuple[list[path], list[Image]]

    Therefore, 'unzip' the input into two lists.

    Parameters:
        batch: list[tuple[str, Image.Image]] - Batch of tuples

    Returns:
        paths: list[str] - A list of paths
        images: list[Image.Image] - A list of corresponding images
    """
    paths = []
    images = []
    for path,image in batch:
        paths.append(path)
        images.append(image)
    return paths, images


def get_datasets() -> dict[str, pd.DataFrame]:
    """
    Gather data from the RFW and BFW dataset, return a dictionary containing dataframes
    which contain the necessary info.
    This function is subject to many changes, and depending on the dataset have different
    DataFrame structures.

    It assumes each DataFrame contains at least:
    `path1`, `path2`: str - The paths to the left and right image
    `same`: int - Indicating if the image pair is of the same person
    `pair`: str - Like same, only 'Genuine' and 'Imposter', rather than 1 and 0 (resp)
    """

    datasets = dict()

    print("Gathering metadata for RFW dataset...")
    # Fist generate RFW dataframe
    df = generate_rfw_df(os.path.join(DATA_ROOT['rfw'], 'txts/'))

    # Some stats
    print("\tFound {} image pairs across {} folds. Dataset has {}/{} pos/neg samples and contains ethnicities {}".format(
        len(df), len(df['fold'].unique()),
        sum(df['same']), sum(df['same']==0),
        set(df['ethnicity'].unique()),
    ))

    # Save intermediate results, contains of images that might not get embedded!
    fname = os.path.join(DATA_ROOT['rfw'], 'rfw_unfiltered.csv')
    df.to_csv(fname, index=False)
    print("\tSaved results to",fname)

    datasets['rfw'] = df


    print("Gathering metadata for BFW dataset...")
    # Can be read directly
    df = pd.read_csv( BFW_RAW_CSV )

    # Rename columns
    df.rename(columns={'p1':'path1', 'p2':'path2', 'label': 'same'}, inplace=True)
    df['pair'] = df['same'].map(lambda x: 'Genuine' if x else 'Imposter')

    # Some stats
    print("\tFound {} image pairs across {} folds. Dataset has {}/{} pos/neg samples and contains attributes {}".format(
        len(df), len(df['fold'].unique()),
        sum(df['same']), sum(df['same']==0),
        set(df['att1']) | set(df['att2']),
    ))

    datasets['bfw'] = df

    return datasets


def get_models() -> dict[str, torch.nn.Module | "ArcFace"]:
    """
    Load models into a dictionary, assumes models are torch modules
    """
    return {
        'facenet': InceptionResnetV1(pretrained='vggface2').eval(),
        'facenet-webface': InceptionResnetV1(pretrained='casia-webface').eval(),
        # TODO: Replace with path to where the model is saved.
        # 'arcface': ArcFace("../arcface_resnet100/resnet100.onnx"),
    }


def get_MTCNN() -> MTCNN:
    """
    Added in case more parameter tuning of MTCNN is required, do so here.
    """
    return MTCNN(device=device)


@torch.no_grad()
def crop(paths: list[str],
         read_root: str,
         save_root: str,
         mtcnn: MTCNN,
         pbar: bool = True,
         batch_size: int = 1,
         resize: tuple[int,int]|None = None
        ) -> list[str]:
    """
    Crop given images and save them to the output directory.

    Does work with batches, but does then require to set a resize size, as MTCNN
    assumes all images in a batch are of same size. If original size is prefered,
    set batch_size to 1 and resize to None.

    Parameters:
        paths: list[str] - A list of paths, starting from read_root
        read_root: str - A common directory of all images, full path of image i is then read_dir+'/'+paths[i]
        save_root: str - What directory to save cropped images in, it is strongly adviced to not re-use read_root
        mtcnn: MTCNN - An MTCNN instance to crop with
        pbar: bool - Whether to use a TQDM progressbar
        batch_size: int - Speaks for itself
        resize: tuple[int,int] - To what size to resize images

    Returns:
        paths: list[str] - A list of paths MTCNN managed to detect a face in
    """

    # Only if youre 100% sure images are of the same size, comment this line out
    assert batch_size == 1 or resize is not None, "Cannot batch without resizing!"

    out = []

    images = PathToImage(
        image_root=read_root,
        paths=paths,
        resize=resize
    )
    paths_loader = DataLoader(
        images,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate
    )

    # Convert loader to TQDM pbar
    if pbar:
        paths_loader = tqdm.tqdm(paths_loader, ncols=100, mininterval=.5, unit='img', unit_scale=batch_size)

    mtcnn.eval()
    for paths,images in paths_loader:
        # Convert to full path
        save_to = [os.path.join(save_root, path) for path in paths]

        # Crop images and save
        cropped_imgs: list[None|torch.Tensor] = mtcnn(images, save_path=save_to)

        # Filter paths where image is saved
        for path,img in zip(paths, cropped_imgs):
            if img is not None:
                out.append(path)

    return out


@torch.no_grad()
def embed(model: torch.nn.Module,
          root: str,
          paths: list[str],
          pbar: bool = True,
          batch_size: int = 1
         ) -> dict[str,np.ndarray]:
    """
    Embed images in batches using provided model. Works similarly to crop method

    Assumes all images are of the same size

    Parameters:
        model: torch.nn.Module - A torch module used to embed
        root: str - A common directory of all paths
        paths: list[str] - A list of paths, starting from root and ending at images
        pbar: bool - Whether to display a TQDM progress bar while embedding
        batch_size: int - Number of items to use in a batch

    Returns:
        embeddings: dict[str, np.ndarray] - A dictionary mapping each item in
        paths to a 2D np array of shape (1,D), where D is the embedding size of
        the provided model.
    """

    # From https://github.com/timesler/facenet-pytorch/blob/master/models/mtcnn.py#L508-L510
    def norm(tensor):
        return (tensor - 127.5) / 128.0

    # Maps paths to embeddings
    embeddings = dict()

    # Set up dataset using PathToImage set, allows to use DataLoaders!
    # Images are read as PIL.Image.Image instances, but model expects batches of
    # torch.Tensor, so add a transform and normalize
    dataset = PathToImage(
        image_root=root,
        paths=paths,
        resize=None,
        transform=transforms.Compose([
            transforms.PILToTensor(),
            norm,
            ])
    )

    # Set up loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate
    )

    # Set model to eval mode on proper device
    model.eval()
    model.to(device)

    # Convert loader to TQDM progresss bar
    if pbar:
        loader = tqdm.tqdm(loader, ncols=100, mininterval=.5, unit='img', unit_scale=batch_size)

    # Actually start embedding
    for paths,images in loader:

        # list[torch.Tensor] -> torch.Tensor
        cropped_imgs = torch.stack(images, dim=0).to(device)

        embs = model(cropped_imgs)

        # Save in dictionary
        for path, emb in zip(paths, embs):
            embeddings[path] = emb.detach().cpu().numpy().reshape(1,-1)

    return embeddings


def similarities(df: pd.DataFrame, embeddings: dict[str, np.ndarray]) -> pd.Series:
    """
    Given a dataframe containing the columns `path1` and `path2`,
    return the cosine similarity of two corresponding embeddings.

    Similarity score is `np.nan` if either `path` has no embedding in `embeddings`.

    Parameters:
        df: pd.DataFrame - A dataframe containing the columns `path1` and `path2`
        embeddings: dict[str, np.ndarray] - A dictionary mapping paths to embeddings

    Returns:
        sim: pd.Series - A Series instance containing similarity scores, either np.nan or in [-1, 1]
    """

    def cos_sim(row: pd.Series) -> float:
        """
        Given a row with columns `path1` and `path2`, compute similarity scores
        using embeddings defined above.
        """
        # Skip unknown embeddings
        if row['path1'] not in embeddings\
        or row['path2'] not in embeddings:
            return np.nan

        # Get embeddings as 1D array
        v1 = np.squeeze(embeddings[row['path1']])
        v2 = np.squeeze(embeddings[row['path2']])

        # Normalize (probably not necessary, but prevents bugs)
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)

        # Cosine similarity
        return np.dot(v1, v2)

    # Apply to `path1`, `path2` columns
    return df[ ['path1', 'path2'] ].apply(cos_sim, axis=1 )


def check_preprocess(conf: Namespace):
    """assert that embeddings exists, etc.
    """
    raise NotImplementedError()


def main():
    # InceptionResnetV1 has a deprecated warning where a list of dictionaries is
    # converted to a `np.ndarray`, surpress these.
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

    # Datasets
    datasets = get_datasets()
    print("Available datasets:", set(datasets.keys()))

    # models
    models = get_models()
    print("Available models:", set(models.keys()))

    # MTCNN
    mtcnn = get_MTCNN()

    print()

    # First crop dataset, then embed
    for dataset in datasets:
        print("Dataset:", dataset)

        df: pd.DataFrame = datasets[dataset]
        paths: list[str] = list( set(df['path1']) | set(df['path2']) )

        print("Cropping images...")
        if SKIP_CROP:
            # Whatever paths can be found in cropped image folder
            cropped_paths = [path for path in paths if os.path.isfile(os.path.join(CROPPED_IMAGE_ROOT[dataset], path))]
            print("\tFound {}/{} existing cropped images".format(len(cropped_paths), len(paths)))
        else:
            # Crop images using MTCNN
            cropped_paths = crop(
                paths=paths,
                read_root=RAW_IMAGE_ROOT[dataset],
                save_root=CROPPED_IMAGE_ROOT[dataset],
                mtcnn=mtcnn,
                pbar=DO_PBAR,
                batch_size=BATCH_SIZE,
                resize=RESIZE
            )
            print("\tCropped {}/{} images".format(len(cropped_paths), len(paths)))

        for modelName in models:
            print(f"\tEmbedding using {modelName} model...")
            model = models[modelName]
            embeddings = embed(
                model=model,
                root=CROPPED_IMAGE_ROOT[dataset],
                paths=cropped_paths,
                pbar=DO_PBAR,
                batch_size=BATCH_SIZE,
            )
            print("\t\tEmbedded {}/{} images".format(len(embeddings), len(paths)))

            # Save embeddings
            fname = os.path.join( EMBEDDING_FOLDER[dataset], EMBEDDING_FORMAT.format(modelName) )
            print(f"\t\tSaving embeddings to {fname}")
            with open(fname, 'wb') as f:
                pickle.dump(embeddings, f)

            print("\tComputing similarity scores...")
            df[modelName] = similarities(df, embeddings)

        model.to(cpu)

        # Drop rows (ie image pairs) that have no embeddings for them
        print("\tDropping un-embedded pairs...")

        # original length
        n = len(df)
        # Drop any image pair where there is no similarity score
        df.dropna(axis='index', subset=list(models.keys()), inplace=True)

        # Some stats
        print("\tDataset {}: keeping {}/{} image pairs".format(dataset, len(df), n))
        print("\tSaving to {}".format(OUTPUT_CSV[dataset]))

        # Output CSV
        df.to_csv(OUTPUT_CSV[dataset], index=False)

    print("Done!")


if __name__ == '__main__':
    main()
