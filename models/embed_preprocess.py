from collections import defaultdict
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
from rfw import RFWImages
from arcface import ArcFace

# Set up device
device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Load models to CPU
print("Loading models...")
models = {
    # 'facenet': InceptionResnetV1(pretrained='vggface2').eval(),
    # 'facenet-webface': InceptionResnetV1(pretrained='casia-webface').eval(),
    # TODO: Replace with link where the model is saved.
    'arcface': ArcFace("../arcface_resnet100/amazon-resnet100.onnx"),
}
print("Available models:", ', '.join(models))

# Crop images using MTCNN
mtcnn = MTCNN(device=device)

# Use BFWImages to load unique images from BFW set
print("Setting up dataset...")
datasets = {
    'bfw': BFWImages(data_root='../data/bfw/uncropped-face-samples/',
                     csv_file='../data/bfw/bfw-v0.1.5-datatable.csv'),
    # 'rfw': RFWImages(data_root='../data/rfw/data/',
    #                  csv_file='../data/rfw/txts/filtered_pairs.csv')
}
print("Available models:", ', '.join(datasets))

# For each model...
for modelName in models:

    print("Using model:", modelName)
    model = models[modelName].to(device)

    for datasetName in datasets:

        print("Using dataset", datasetName)
        dataset = datasets[datasetName]

        # Save embeddings here
        embeddings: "dict[str, np.ndarray]" = defaultdict(dict)

        # Iterate images
        # LIMIT = None
        LIMIT = 5_000
        if LIMIT is None:
            iterator = enumerate(dataset)
        else:
            iterator = zip(range(LIMIT), dataset)

        for _i, (img, meta) in tqdm.tqdm(iterator, ncols=100):

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
            emb = emb.detach().cpu().numpy()

            # Save embedding to CPU
            embeddings['full'][meta['path']] = emb
            embeddings[meta['ethnicity']][meta['path']] = emb

        # Save embeddings to pickle file
        for subset in embeddings:
            embeddings_subset = embeddings[subset]
            save_at = "/".join(dataset.data_root.split("/")[:-1])
            if subset == 'full':
                fname = os.path.join(save_at, f'{modelName}_embeddings.pickle')
            else:
                fname = os.path.join(save_at, f'{subset}_{modelName}_embeddings.pickle')

            print("Mapping path->embedding saved to", fname)
            with open(fname, 'wb') as file:
                pickle.dump(embeddings_subset, file)
