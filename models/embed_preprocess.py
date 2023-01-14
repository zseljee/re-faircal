import numpy as np
import pandas as pd
import os
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import Sequential

from facenet_pytorch import InceptionResnetV1
from bfw import BFWImages

device = torch.device( 'cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

print("Loading models...")
models = {
    'facenet_vggface2': InceptionResnetV1(pretrained='vggface2'),
    'facenet_webface': InceptionResnetV1(pretrained='casia-webface')
}
print("Available models:", ', '.join(models))

print("Setting up dataset...")
transform = transforms.Compose([
                transforms.ToTensor(),
            ])
bfw = BFWImages(data_root='../data/bfw/uncropped-face-samples/',
                csv_file='../data/bfw/bfw-v0.1.5-datatable.csv',
                transform=transform)
loader = DataLoader(bfw, batch_size=1, shuffle=False)

    
for modelName in models:
    print("Using model:", modelName)
    model = models[modelName].eval().to(device)

    embeddings = []
    for samples in tqdm.tqdm(loader):

        imgs = samples[0]

        imgs = imgs.to(device)
        try:
            emb = model(imgs)
            emb = emb.detach().cpu()
            embeddings.append(emb)
        except:
            pass

    embeddings = torch.concat( embeddings, dim=0 )
    print("Embeddings shape:", embeddings.shape)
