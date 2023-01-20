"""
Preprocess the images of the

Author: Zirk Seljee
Email: zirk.seljee@student.uva.nl
Affiliation: University of Amsterdam
"""
# TODO: Speed up the calculations of MTCNN, possibly with GPU or batching.
# TODO: Combine this with @Simon's RFW filtering method and my BFW filtering?

import argparse
import datetime  # For getting rough ideas of how long things take
import os
from pathlib import Path
from PIL import Image
from typing import Any, Callable, Iterator, Literal, TypeAlias

import torch

# Importing this takes long, can we speed up this so that the argparse is done
# quicker with parsing and you don't have to wait 5 seconds until you see that
# you missed something from it.
from facenet_pytorch import MTCNN


PairType: TypeAlias = tuple[str, int, str, int]
FoldType: TypeAlias = tuple[list[PairType], list[PairType]]

ALL_DATASETS = ["BFW", "RFW"]
DatasetType: TypeAlias = Literal["BFW", "RSW"]

ALL_STEPS = ["MTCNN", "unrecognised", "filter"]

VERBOSITY = 0
"""How much information should be printed?
0: Nothing except problems and statistics when done.
1: Some minor statistics about progress.
2: All progress that you may want to monitor if you have to check reproducability.
3: As much as reasonably possible, like what choices are made for surprissing if-statements.
"""

##############################################################################
###  This was previously useful, maybe not in this same way atm, maybe     ###
###  needed for verifying procided pairs                                   ###
##############################################################################

def all_rfw_folds(datafolder: Path):
    """Return a dictionary mapping all subsets to a list of folds of
    positive and negative example pairs.

    Returns: fold - List of (positive, negative) pairs both with (folder, photo, folder, photo) ordering for possibly chaining lists later
    """
    all_folds: dict[str, list[FoldType]] = dict()
    for subset in ["African", "Asian", "Caucasian", "Indian"]:
        pair_list = datafolder.joinpath("txts", subset, f"{subset}_pairs.txt")
        with open(pair_list, "r") as f:
            # List of (positive, negative) pairs both with (folder, photo, folder, photo) ordering for possibly chaining lists later
            folds: list[FoldType] = []
            # There are 10 different folds defined in the files.
            for i in range(10):
                # Positive examples
                positives = []
                for j in range(300):
                    identity_folder, photo_1, photo_2 = f.readline().rstrip().split("\t")
                    positives.append((identity_folder, int(photo_1), identity_folder, int(photo_2)))

                # Negative examples
                negatives = []
                for j in range(300):
                    identity_folder_1, photo_1, identity_folder_2, photo_2 = f.readline().rstrip().split("\t")
                    negatives.append((identity_folder_1, int(photo_1), identity_folder_2, int(photo_2)))

                folds.append((positives, negatives))
            all_folds[subset] = folds
    return all_folds


# TODO: Make this reversable to the pairs as well, porbably with metadata
def folds_to_pairs(folds: dict[str, list[FoldType]]) -> list[PairType]:
    """Converts a dictionary of folds to a list of pairs."""
    all_pairs = []
    for fold in folds.values():
        for pos, neg in fold:
            all_pairs.extend(pos)
            all_pairs.extend(neg)
    return all_pairs


get_all_pairs: dict[DatasetType, Callable[[Path], list[PairType]]] = {
    "BFW": NotImplementedError,
    "RFW": lambda x: folds_to_pairs(all_rfw_folds(x)),
}

##############################################################################
###  End of (temporarily) unused code                                      ###
##############################################################################


def get_datafolder(args: argparse.Namespace, dataset: DatasetType) -> Path:
    """Return the datafolder for the dataset, according to the commandline arguments."""
    path = getattr(args, f"{dataset.lower()}_datafolder")
    return path


def get_dataset_folders(args: argparse.Namespace, dataset: DatasetType) -> tuple[Path, Path, Path]:
    """Return the paths to the imagefolder of the dataset, where to save the
    cropped images and where to find/save the text files.
    """
    path = get_datafolder(args, dataset)
    if dataset == "BFW":
        original_images_path = path.joinpath("uncropped-face-samples")
        cropped_images_path = path.joinpath("cropped-face-samples")
        txts_path = path
    elif dataset == "RFW":
        original_images_path = path.joinpath("data")
        cropped_images_path = path.joinpath("cropped_data")
        txts_path = path.joinpath("txts")
    else:
        raise ValueError("Unrecognised dataset")

    # Create a folder to store the cropped faces in case this is the first time
    if VERBOSITY >= 1 and not cropped_images_path.exists():
        print("Creating folder to store cropped images at", cropped_images_path)
        # Since we tested existence anyway to print the creation, we don't technically need exists_ok,
        # unless someone created the folder themselves in the meantime (unlikely)
        cropped_images_path.mkdir(parents=False, exist_ok=True)

    return original_images_path, cropped_images_path, txts_path


def get_RFW_or_BFW_walk(data_folder) -> Iterator[Path]:
    """Returns an iterator across all relative paths to photos in the RFW dataset."""
    walker = os.walk(data_folder)
    top_directory, dirnames, filenames = next(walker)
    yield from [filename for filename in filenames if filename.endswith(".jpg")]  # Should be empty
    # Go through all sub-directories and collect all files
    for dirpath, dirnames, filenames in walker:
        dirpath: str  # typing hint
        removed_dir = dirpath.removeprefix(top_directory).lstrip("/")
        assert removed_dir != dirpath, "Directory has a different head than the top-level, possibly symlink problems? Or just always impossible."
        # Create relative paths to the top-directory for all files in this folder
        for filename in filenames:
            # Skip non-image files
            if not filename.endswith(".jpg"):
                if VERBOSITY >= 3:
                    print(f"Skipping file {removed_dir}/{filename}")
                continue
            yield Path(removed_dir).joinpath(filename)


def get_dir_walk(dataset: DatasetType, data_folder: Path):
    """Returns an iterator across all relative paths to photos in a specific dataset.

    The reason for this function was that I don't know if we can use the same
    naive walk for all datasets.  (I would assume so, though.)
    """
    if dataset == "RFW":
        yield from get_RFW_or_BFW_walk(data_folder)
    elif dataset == "BFW":
        yield from get_RFW_or_BFW_walk(data_folder)
    else:
        raise ValueError("Unknown dataset")


def do_MTCNN_cropping(args) -> dict[str, Any]:
    """Crop all images of all given datasets.

    Does not actually creates a CSV files of `identities,image,keptBool`,
    that was the idea initially though.  I don't know what the output of
    mtcnn would be when it cannot recognise a face.  Test it with a known
    unrecognised face to speed up the creation of the csv (although it
    didn't take close to anywhere as much time as running mtcnn itself.)

    Returns a dictionary with information about the process.
    """
    images_processed = 0
    failed_images = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(device=device)
    for dataset in args.datasets:
        if VERBOSITY >= 1:
            print(f"\tStarting dataset {dataset}.")

        original_images_path, cropped_images_path, _txts_path = get_dataset_folders(args, dataset)

        # Do dataset specific walk through photos
        for relative_photo_path in get_dir_walk(dataset, original_images_path):
            if VERBOSITY >= 3:
                print("\t\tRunning MTCNN on image", relative_photo_path)
            original_photo_path = original_images_path.joinpath(relative_photo_path)
            cropped_photo_path = cropped_images_path.joinpath(relative_photo_path)
            cropped_photo_path.parent.mkdir(parents=True, exist_ok=True)
            img = Image.open(original_photo_path)
            # Specific parameters for the MTCNN that was used are unknown.
            # Wrap the save_path in a list because Path objects are not strings and that's the only internal check to separate
            # this input from an input on multiple images.
            # https://github.com/timesler/facenet-pytorch/blob/2633b2daeb0f93fb1758a46c76c2acd6946a2e13/models/mtcnn.py#L468-L469
            out = mtcnn(img, save_path=[cropped_photo_path])
            if out is None:
                failed_images.append(relative_photo_path)
                if VERBOSITY >= 3:
                    print("\t\t  (Didn't recognise a face)")
            images_processed += 1

        if VERBOSITY >= 1:
            print("\t  Done!")

    return {
        "images_processed": images_processed,
        "failed_images": failed_images,
    }


def find_lost_pairs(args) -> dict[str, Any]:
    """Create a file with each line an identity and photo that is not
    recognised by MTCNN.

    The file is save as `unrecognised-faces.txt` and savd in the txts
    folder for RFW.

    Returns a dictionary with information about the process.
    """
    for dataset in args.datasets:
        original_images_path, cropped_images_path, txts_path = get_dataset_folders(args, dataset)

        with open(txts_path.joinpath("unrecognised-faces.txt"), "w") as outfile:
            # Do dataset specific walk through photos
            for relative_photo_path in get_dir_walk(dataset, original_images_path):
                original_photo_path = original_images_path.joinpath(relative_photo_path)
                cropped_photo_path = cropped_images_path.joinpath(relative_photo_path)
                # If a photo only exists in the original dataset, assume a face wasn't found
                if original_photo_path.is_file() and not cropped_photo_path.is_file():
                    outfile.write(str(relative_photo_path) + "\n")


def do_pair_filtering(args) -> dict[str, Any]:
    """@Simon did this in the filter/filter_images.py file.
    I think that code only works on the RFW dataset, though.

    Did @Jip also do this on the BFW dataset in the models/embed_preprocess.py file?
    """
    raise NotImplementedError()
    for dataset in args.datasets:
        path = get_datafolder(args, dataset)
        pairs = get_all_pairs[dataset](path)


def main() -> None:
    args = get_args()
    args = validate_args(args)

    info_dict = {}

    # Do preprocessing
    if "MTCNN" in args.steps:
        if VERBOSITY >= 1:
            print("Doing MTCNN cropping...")
        info_dict.update(do_MTCNN_cropping(args))
        info_dict["num_mtcnn_unrecognised"] = len(info_dict["failed_images"])
        # Delete because we don't want to clutter printing the results...
        # We're creating the list from scratch anyway in the next step.
        del info_dict["failed_images"]
    elif VERBOSITY >= 1:
        print("Skipping MTCNN cropping.")

    if "unrecognised" in args.steps:
        if VERBOSITY >= 1:
            print("Finding all images that did not have a face recognised by MTCNN...")
        info_dict.update(find_lost_pairs(args))
    elif VERBOSITY >= 1:
        print("Assuming unrecognised-faces.txt exists.")

    if "filter" in args.steps:
        if VERBOSITY >= 1:
            print("Doing pair filtering...")
        info_dict.update(do_pair_filtering(args))
    elif VERBOSITY >= 1:
        print("Skipping pair filtering.")

    print(info_dict)


def get_args() -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a list of pairs to include and exclude for the given dataset(s).",
    )

    parser.add_argument("-v", "--verbose",
                        action="count",
                        default=0,
                        help="Provide extra information on what process is happening.  Can be stacked up to 4 times.",
    )

    parser.add_argument("-d", "--datasets",
                        choices=ALL_DATASETS + ["all"],
                        default=["all"],
                        help="For which datasets to run the preprocessing",
                        nargs="+",
                        type=str,
    )

    # Where to find the datasets on disk
    for dataset in ALL_DATASETS:
        parser.add_argument(f"--{dataset.lower()}-datafolder",
                            help=f"Path to the {dataset} datafolder from the current working directory.",
                            type=Path,
        )

    # Split up preprocessing in digestable chunks
    # TODO: Come up with more descriptive names for the steps.
    parser.add_argument("-s", "--steps",
                        choices=ALL_STEPS + ["all"],
                        default=["all"],
                        help="The different preprocessing steps that should be executed.",
                        nargs="+",
                        type=str,
    )

    args = parser.parse_args()
    return args


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    """Process the arguments and check they all have appropriate values."""
    global VERBOSITY
    VERBOSITY = args.verbose

    # Extract all datasets
    if "all" in args.datasets:
        args.datasets = ALL_DATASETS
        if VERBOSITY >= 3:
            print(f"Using `all` as datasets: replacing with {repr(ALL_DATASETS)}")

    # Assert that all used datasets are actually known
    for dataset in args.datasets:
        assert get_datafolder(args, dataset) is not None, f"Please provide a datapath to the {dataset} folder!"

    # Extract all steps
    if "all" in args.steps:
        args.steps = ALL_STEPS
        if VERBOSITY >= 3:
            print(f"Using `all` as steps: replacing with {repr(ALL_STEPS)}")

    return args


if __name__ == "__main__":
    main()
