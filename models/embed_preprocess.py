import numpy as np
import pandas as pd
import os
import tqdm
import pickle

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from facenet_pytorch import InceptionResnetV1, MTCNN
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

mtcnn = MTCNN(device=device, thresholds=[.95, .95, .95])

# Use BFWImages to load unique images from BFW set
print("Setting up dataset...")
transform = transforms.Compose([
                # transforms.ToTensor(),
            ])
bfw = BFWImages(data_root=DATA_ROOT,
                csv_file='../data/bfw/bfw-v0.1.5-datatable.csv',
                transform=transform)

# !IMPORTANT! Use batch size 1 as images are of varying size
loader = DataLoader(bfw, batch_size=None, shuffle=False)

# For each model...
for modelName in models:

    print("Using model:", modelName)
    model = models[modelName].to(device)

    # Save embeddings here
    embeddings: dict[str, np.ndarray] = dict()

    # Iterate images
    for img, meta in tqdm.tqdm(loader, ncols=100):

        # Crop image using MTCNN
        cropped_img: torch.Tensor = mtcnn(img)

        # No face detected (or at least not with the given threshold)
        if cropped_img is None:
            # TODO what to do here?
            continue

        # 'batch' it as a single image, model expects 4D input
        cropped_img = cropped_img.to(device).unsqueeze(dim=0)

        # Create embedding
        emb = model(cropped_img)

        # Save embedding to CPU
        embeddings[meta['path']] = emb.detach().cpu().numpy()

    # Save embeddings to pickle file
    fname = os.path.join(DATA_ROOT, f'{modelName}_embeddings_thrhld095.pickle')
    print("Mapping path->embedding saved to", fname)
    with open(fname, 'wb') as file:
        pickle.dump(embeddings, file)