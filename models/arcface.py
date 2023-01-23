"""

Based on the Pytorch tutorial on how to use onnxruntime:
https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
"""

import numpy as np
import onnx
import torch

from onnxruntime import InferenceSession
from torchvision.transforms import Resize


class ArcFace():
    """Wrapper function for the onnx model and its runtime to handle it
    as a pytorch like object.
    """
    def __init__(self, model_path: str):
        onnx_model = onnx.load_model(model_path)
        # Sanity check that the model works
        onnx.checker.check_model(onnx_model)
        self.onnx_model = onnx_model
        self.ort_session = InferenceSession(model_path, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])

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
            resized_img = Resize((112, 112))(img).unsqueeze(0)
            # Compute ONNX Runtime output prediction
            # print(resized_img.shape)
            ort_session = self.ort_session
            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(resized_img)}
            ort_outs = ort_session.run(None, ort_inputs)
            emb_batch.append(ort_outs[0])
        return torch.tensor(np.array(emb_batch))

    def to(self, _device):
        """Eats the call to move the model to the GPU, no clue how to do
        that with onnxruntime in a way that's compatible with pytorch.
        """
        return self


# Utility function
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
