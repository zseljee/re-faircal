"""
Preprocess the images of the

Author: Zirk Seljee
Email: zirk.seljee@student.uva.nl
Affiliation: University of Amsterdam
"""

import argparse
from pathlib import Path
from typing import Callable, Literal, TypeAlias

# from mtcnn import MTCNN


PairType: TypeAlias = tuple[str, int, str, int]
FoldType = tuple[list[PairType], list[PairType]]

ALL_DATASETS = ["BFW", "RFW"]

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


get_all_pairs: dict[Literal["BFW", "RSW"], Callable[[Path], list[PairType]]] = {
    "BFW": NotImplementedError,
    "RFW": lambda x: folds_to_pairs(all_rfw_folds(x)),
}


def main() -> None:
    args = get_args()
    args = validate_args(args)

    # Do preprocessing
    # mtcnn = MTCNN()
    for dataset in args.datasets:
        path = getattr(args, f"{dataset.lower()}_datafolder")
        pairs = get_all_pairs[dataset](path)

        print(pairs[295:305])


def get_args() -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a list of pairs to include and exclude for the given dataset(s).",
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

    args = parser.parse_args()
    return args


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    """Process the arguments and check they all have appropriate values."""
    # Extract all datasets
    if "all" in args.datasets:
        args.datasets = ALL_DATASETS

    print(args)
    
    # Assert that all used datasets are actually known
    for dataset in ALL_DATASETS:
        if dataset in args.datasets:
            assert getattr(args, f"{dataset.lower()}_datafolder") is not None, f"Please provide a datapath to the {dataset} folder!"
    
    return args


if __name__ == "__main__":
    main()
