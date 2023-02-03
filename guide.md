# Step by step guide

This file describes the step-by-step process to reproduce the results from our reproducibility paper.  For a quickstart, check out the main readme file.

## Requirements

Before we start, there are a few pre-requisities that you should have before we can clone the repository and get started.  First, make sure you are on a Unix based system (Linux or MacOS).  Next, make sure you have `conda` installed.  If you don't install miniconda using the guide [here][miniconda].

## Setting up the environment

To get started, clone this repository in the ordinary way you're used to, e.g. with `git clone`:

```sh
git clone git@github.com:zseljee/re-faircal.git [optional diffferent foldername]
cd <repo name OR custom foldername>
```

Now that you have the code installed locally, install either the CPU or GPU based conda environments, whichever is appropriate for you system.  The default name of this environment is `mlrc-faircal`, but you can specify a different name to use with the option `-n`.  To install the environment, use:

```sh
conda env create -f environment_<cpu|gpu>].yml [-n other-env-name]
```

While we did not succeed in getting appropriate embeddings for the ArcFace model, we have a setup that supposedly creates the correct embeddings.  This requires the use of `mxnet`, which requires a lower version of Python.  For this reason, we added an additional environment in the file `environment_cpu_arcface.yml` that additionally installs an environment named `mlrc-faircal-arcface`.

## Data

Now that installing the environment hopefully succeeded, you should retrieve the data from their original sources, which you can find linked in the paper and the main readme.  Although we explain the expected structure below, it is possible to use a different structure as all files contain command-line arguments for the data location.

### BFW

This dataset is described in the [data readme].  We expect the folder with `jpg`s of the uncropped faces to be put in the folder `data/bfw/uncropped-face-samples/`.  The csv file describing the different folds of the data should be stored under `data/bfw/bfw-v0.1.5-datatable.csv`.

### RFW

This dataset is also described in the [data readme].  We expect the test folder you can download to be renamed `rfw` and put at the same level as the BFW folder.

## Preprocessing

Preprocessing is done automatically when the experiments recognise they are being run for the first time.

## Experiments

The experiments can be run in their entirety with the command below.

```sh
python3 src/main.py
```

However, this code may fail if no ArcFace embeddings are present depending on the specific set-up chosen.  If this is the case, change environment to create the ArcFace embeddings before going back to run the experiments.

## Results (Figures and Tables)

To create figures and tables, you can run the final notebook `src/Tables and Figures.ipynb` or run the following 2 python files:

```sh
python3 src/tables_and_figures.py
python src/extension.py
```

[data readme]: ./data/README.md
[miniconda]: https://docs.conda.io/en/latest/miniconda.html
