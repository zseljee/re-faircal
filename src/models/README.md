# Pre-processing of data
*Only available for BFW and RFW datasets*

## Obtaining raw data

Please see the [data](../../data/README.md) readme for information on obtaining the original data and file structures. This module assumes this file strucute is exactly as described there.

Furthermore, please see the `constants.py` file as to where some files are expected to be read from or saved to.

## Obtaining pre-trained models

Embedding the images from the dataset is done using two Inception ResNet models, one trained on the VGGFace2 daatset and another on the CASIA-Webface dataset.
The PyTorch implementation and pre-trained models can be found [here](https://github.com/timesler/facenet-pytorch).
Installing the `facenet-pytorch` module is also a requirement to run compute the embeddings.
Installing this package should install the pre-trained models in the PyTorch cache. It is therefore not necessary to download these models manually.

## Setting up for embedding

### BFW
The BFW set already contains a convenient file (`bfw-v<version>-datatable.csv`) for getting image paths, which will be used throughout the pre-processing.
In order to obtain the image paths, the set of the `path1` and `path2` columns is taken.

### RFW
The RFW has a more complicated structure, where the information of all image pairs, file paths and folds is spread ocross several files.
In order to parse this, the code in `gen_rfw_table.py` has been provided, which walks through the `txts` folder, looking for `txts/<set>/<set>_pairs.txt` files.
These `pairs.txt` file contain the necessary information to generate a dataframe with the columns `id1`,`id2`,`path1`,`path2`,`label`,`fold`,`ethnicity`,`num1`,`num2`.

* **id1**,**id2** is an identifier for a person within a certain subgroup. This ID is however not unique across ethnicities.
* **path1**, **path2** is the path to the first common folder of all images. This can be seen as the data root of this dataset.
* **label** is either 0 or 1, indicated a False or True image pair resp.
* **fold** is an integer in the range $1, ..., K$, where $K$ is the number of folds (10 for this dataset).
* **ethnicity** is the ethnicity of this image pair, the BFW dataset contains only same-ethnicity pairs.
* **num1**, **num2** indicates which image of the person has be used.


## Cropping Images
After setting up the `.csv` files, the raw image files need to be cropped.
A set of paths is taken using the now available dataframes, which are then opened and fed through a MTCNN model, as is common practice.
This MTCNN uses default parameters, and saves images to a new folder for each dataset, as to save computation when embedding using multiple models.
MTCNN might not detect all possible faces, so it is important to make sure to filter the list of paths between this step and the embeddings accordingly.

## Embedding Images
Reading the now cropped images, embed any image that has been parsed by the MTCNN.
In order to keep the mapping from path to embedding, save the results as a dictionary mapping paths to `np.ndarray`s.
Finally, save the entire dictionary as a `.pickle` file, which will be read from in the main part of this project.
