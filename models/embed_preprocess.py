import numpy as np
import pandas as pd
import os
import tqdm
import pickle

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from facenet_pytorch import InceptionResnetV1
from bfw import BFWImages

DATA_ROOT = os.path.abspath('../data/bfw/uncropped-face-samples/')

# Set up device
device = torch.device( 'cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Load models to CPU
print("Loading models...")
models = {
    'facenet': InceptionResnetV1(pretrained='vggface2').eval(),
    'facenet-webface': InceptionResnetV1(pretrained='casia-webface').eval(),
    # 'arcface': NotImplementedError(),
}
print("Available models:", ', '.join(models))

# Use BFWImages to load unique images from BFW set
print("Setting up dataset...")
transform = transforms.Compose([
                transforms.ToTensor(),
            ])
bfw = BFWImages(data_root=DATA_ROOT,
                csv_file='../data/bfw/bfw-v0.1.5-datatable.csv',
                transform=transform)

# !IMPORTANT! Use batch size 1 as images are of varying size
loader = DataLoader(bfw, batch_size=1, shuffle=False)

# For each model...
for modelName in models:

    print("Using model:", modelName)
    model = models[modelName].to(device)

    # Save embeddings here
    embeddings: dict[str, np.ndarray] = dict()

    # Iterate images
    for img, meta in tqdm.tqdm(loader):

        img = img.to(device)

        # Some images are of invalid size
        try:
            # Create embedding
            emb = model(img)

            # Save embedding to CPU
            embeddings[meta['path']] = emb.detach().cpu().numpy()
        except:
            # TODO: set embedding to None or don't add at all?
            # embeddings[meta['path']] = None
            pass

    # Save embeddings
    fname = os.path.join(DATA_ROOT, f'{modelName}_embeddings.pickle')
    print("Mapping path->embedding index saved to", fname)
    with open(fname, 'wb') as file:
        pickle.dump(embeddings, file)