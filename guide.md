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

Some of the preprocessing steps may be skipped, as they are only necessary for investigating if the intermediate results are as expected.

### Store MTCNN results

For creating the embeddings, there is a minor difference between using the immediate output of the MTCNN network or first saving this as files and then loading these files.  You may also wish to store the MTCNN output for inspection.  The following command creates subdirectories in the same structure for each dataset with the cropped images.  Faces that aren't recognised will be missing in the new directory.  Specifically, the BFW dataset will be stored under `[data/bfw/]data_cropped/` and RFW under `[data/rfw/]data_cropped`.

```sh
python3 src/mtcnn_preprocess.py -v[vvv] --dataset all [BFW] [RFW] --steps MTCNN unrecognised --bfw_datafolder <path> --rfw_datafolder <path>
```

### Filter the embeddings

From the dataset created by the MTCNN, we can find out which pairs should be excluded from our training.  This step recreates the tables that are necessary with only information from the pairs that can be used.

```sh
python3 filter/filter_images.py
```

### Creating the embeddings

This requires a file from the previous step, which requires also running the MTCNN on the RFW dataset.  Although we store the resulting crops, we do not use them, because we decided the direct output of the MTCNN would be more accurate than the output re-read from the stored `jpg` files due to the compression this does.

```sh
cd models
python3 embed_preprocess.py
cd ..
```

## Experiments

The experiments can be run in their entirety once the embeddings have been created with the command below.  This specifically has to do the K-means clustering before it can do FairCal, but once the clusters have been created re-running the program gives the results quicker.

```sh
python3 refactored-code/main.py
```

## Results (Figures and Tables)

To create figures and tables, you can run the final program:

```sh
python3 refactored-code/tables_and_figures.py
```

[data readme]: ./data/README.md
[miniconda]: https://docs.conda.io/en/latest/miniconda.html
