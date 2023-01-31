import numpy as np
import os
import pickle
import traceback

from argparse import Namespace

from approaches import uncalibrated, baseline, faircal, oracle, fsn
from approaches.utils import get_metrics
from args import args
from constants import *
from dataset import Dataset
from utils import iterate_configurations, get_experiment_folder


APPROACHES = {
    'uncalibrated': uncalibrated,
    'baseline': baseline,
    'faircal': faircal,
    'oracle': oracle,
    'fsn': fsn,
}


def gather_results(dataset: Dataset,
                   conf: Namespace
                  ) -> dict[str, dict[str, any]]:
    """
    Given a dataset and some configuration, run an experiment on each fold
    of the dataset.

    This method iterates over the Dataset folds and calls the callable in
    APPROACHES.

    The output will be a dictionary containing keys 'fold{K}' with K being
    each fold in the dataset. The values are again dictionaries, containing
    the key 'scores' and 'metrics'. The former is a list/nd.ndarray of the same
    length as the *entire* dataset, indicating the calibrated score. The latter
    is another dictionary with metrics/meta data about the calibrated scores.
    These keys/values can be found in `utils.get_metrics`.

    Parameters:
        dataset: Dataset - A Dataset instance with some number of folds
        conf: Namespace - A argparse.Namespace, just a fancy dictionary containing
        information on the current experiment, such as 'n_cluster'
    
    Returns:
        data: dict[str, dict[str, any]]
    """

    data = {}

    for k in np.copy(dataset.folds):
        print(f"\nFold {k}", '~'*60)

        dataset.set_fold(k)
        calibrated_scores = APPROACHES[conf.approach](dataset=dataset, conf=conf)

        data[f'fold{k}'] = {
            'scores': calibrated_scores,
            'metrics': get_metrics(calibrated_scores, dataset, conf)
        }

    return data


def main():

    os.makedirs(os.path.join(EXPERIMENT_FOLDER, 'kmeans'), exist_ok=True)

    # Try each configuration, as derived from args
    for conf in iterate_configurations(args):

        print("\n"+("="*80))
        print("Running on configuration", conf)

        if (conf.dataset == 'rfw' and conf.feature == 'arcface')\
        or (conf.dataset == 'bfw' and conf.feature == 'facenet'):
            print("Skipping experiment! ArcFace cannot be combined with RFW and FaceNet(VGGFace2) cannot be combined with BFW.")
            continue

        # Save results of the experiment in this folder
        exp_folder = get_experiment_folder(conf)

        dataset = Dataset(name=conf.dataset, feature=conf.feature)

        # Check if experiment is already run
        saveto = os.path.join( exp_folder, 'results.npy' )
        if not os.path.isfile(saveto) or args.ignore_existing:

            try:
                data = gather_results(dataset=dataset, conf=conf)

                with open(saveto, 'wb') as f:
                    pickle.dump(data, f)
            except Exception as e:
                data = None
                print("ERROR, could not run experiment! It gives the following error:")
                print(e)
                traceback.print_exc(limit=None, file=None, chain=True)

        print("\nExperiment finished, find results at", saveto)
        print(("="*80))

    print("Done!")


if __name__ == '__main__':
    main()
