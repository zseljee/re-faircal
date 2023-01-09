>📋  A **partially filled out** template README.md for code accompanying a Machine Learning paper.

# My Paper Title

This repository is the official implementation of [My Paper Title](https://arxiv.org/abs/2030.12345). Based on the paper of [Salvador (2022)](https://canvas.uva.nl/courses/32257/files/8008473?module_item_id=1553026)

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials.

## Requirements

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

To install requirements, first download [miniconda](https://docs.conda.io/en/latest/miniconda.html).  Then, install the appropriate environment by executing the command below from the current directory.  This will create an environment named `fact-ai`.

```setup
conda env create -f environment[_lock]_[cpu|gpu].yml
```

Download the data [here](#).

## Training

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

## Evaluation

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

## Pre-trained Models

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z.

## Results

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it.

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

## Contributing

To edit the paper, go to [this overleaf project](https://www.overleaf.com/project/63bbfc495280d6de78aa8d20) if you have permission, or view the paper with [this link](https://www.overleaf.com/read/zpqvnmvcbgsc).

>📋  Pick a licence and describe how to contribute to your code repository.
