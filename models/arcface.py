"""
Requires the model to be downloaded from:
https://s3.amazonaws.com/onnx-model-zoo/arcface/resnet100.onnx

Based on the notebook on how to run the model:
https://github.com/onnx/models/blob/8e893eb39b131f6d3970be6ebd525327d3df34ea/vision/body_analysis/arcface/dependencies/arcface_inference.ipynb
"""

import mxnet as mx
import numpy as np
import sklearn
import sklearn.preprocessing
import torch

from mxnet.contrib.onnx.onnx2mx.import_model import import_model
from torchvision.transforms import Resize


def get_model(ctx, model):
    image_size = (112,112)
    # Import ONNX model
    sym, arg_params, aux_params = import_model(model)
    # Define and binds parameters to the network
    model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    return model


# Determine and set context
if len(mx.test_utils.list_gpus())==0:
    ctx = mx.cpu()
else:
    ctx = mx.gpu(0)


def get_feature(model, aligned):
    input_blob = np.expand_dims(aligned, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    model.forward(db, is_train=False)
    embedding = model.get_outputs()[0]
    try:
        embedding.asnumpy()
    except:
        pass
    embedding = embedding.asnumpy()
    embedding = sklearn.preprocessing.normalize(embedding).flatten()
    return embedding


class ArcFace():
    """Wrapper function for the onnx model and its runtime to handle it
    as a pytorch like object.
    """
    def __init__(self, model_path: str):
        self.mxnet_model = get_model(ctx, model_path)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, img_batch):
        """Creates an embedding for every image in the batch.

        Actually computes the embeddings for every image separately, as
        that is required by the onnx runtime specification, apparently.

        Converts all images into 112x112 size before running them
        through the network.
        """
        emb_batch = []
        for img in img_batch:
            # The model seems to require images in the range 0, 255
            img = img - img.min()
            img = img * (255 / img.max())
            resized_img = Resize((112, 112))(img)
            emb = get_feature(self.mxnet_model, resized_img)
            emb_batch.append(emb)
        return torch.tensor(np.array(emb_batch))

    def to(self, _device):
        """Eats the call to move the model to the GPU, no clue how to do
        that with mxnet in a way that's compatible with pytorch.
        """
        print("Warning: Cannot change context, not implemented.  Current context is", ctx)
        return self


# Utility function
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
