import numpy as np
import pandas as pd
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from facenet_pytorch import MTCNN, InceptionResnetV1

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    # thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)


dataset = datasets.ImageFolder(f'./data/bfw/uncropped-face-samples/asian_females/n000009/')
print(dataset.class_to_idx)
# dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
# loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)