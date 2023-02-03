# [Re] Reproducibility study of FairCal

This reposotiry tries to reproduce results of [Salvador (2022)](https://openreview.net/forum?id=nRj0NcmSuxb) using an adaption of [the papers code on github](https://github.com/tiagosalvador/faircal).

## Requirements

To install requirements, first download [miniconda](https://docs.conda.io/en/latest/miniconda.html).  Then, install the appropriate environment by executing the command below from the current directory.  This will create an environment named `mlrc-faircal`.

```setup
conda env create -f environment_[cpu|gpu].yml
```

## Data

The datasets used were the BFW and RFW for verifying the original paper results.  Other datasets may be used for additional verification.  See the Table below on where to find the datasets.

| Dataset | Open-source | URL |
|---------|-------------|-----|
| BFW     | No, register through Google Forms | https://github.com/visionjo/facerec-bias-bfw |
| RFW     | No, mail for research access | http://whdeng.cn/RFW/testing.html |

After obtaining these dataset, please read the [Data README](./data/README.md) on how to crop and embed the face images.

## Experiments

Running
```
$ python src/main.py
```
should execute the entire pipeline (crop, embed, cluster, calibrate, evaluate) of this project and save the results in the [experiments folder](./src/experiments/).
In order to generate the tables and figures used in the reproducability paper, please run
```
$ python src/tables_and_figures.py
```

## Pre-trained Models

In order to embed the images, Inception FaceNet models are used, which were obtained from the [FaceNet PyTorch GitHub](https://github.com/timesler/facenet-pytorch).
This will automatically be downloaded using the enviornment defined [above](#requirements).

## Contributing

To edit the paper, go to [this overleaf project](https://www.overleaf.com/project/63bbfc495280d6de78aa8d20) if you have permission, or view the paper with [this link](https://www.overleaf.com/read/zpqvnmvcbgsc).
