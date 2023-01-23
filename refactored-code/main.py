import os
import pickle
import traceback
from typing import Any, Iterator
from argparse import Namespace
import itertools

import numpy as np

from args import args
from constants import *
from dataset import Dataset

from approaches import uncalibrated, baseline, faircal, oracle
from approaches.utils import get_metrics
from visualisations import violinplot, fpr2globalfpr

APPROACHES = {
    'baseline': baseline,
    'faircal': faircal,
    'oracle': oracle,
    'uncalibrated': uncalibrated
}

def iterate_configurations() -> Namespace:
    """
    Iterate options in args to yield configurations.
    Prevents many 'for' loops in main

    Yields Namespace with keys as defined below
    """

    # Keys used in returned Namespace
    keys = ['dataset',
            'n_cluster',
            'feature',
            'approach',
            'calibration_method',
           ]

    # Use those keys to get a list of values for each option
    # ! IMPORTANT ! this assumes that these attributes are lists/iterables
    values = [getattr(args, key) for key in keys]

    # To stress the above point, check if `values` is a list of lists
    assert all(isinstance(val, list) for val in values), "Configuration contains non-list key"

    # Now combine each item in each list of options in values with each other
    # Ie [[1,2], ['a', 'b']] gives (1,a), (1,b), (2,a), (2,b)
    for conf in itertools.product(*values):

        # Now map the keys back to the values
        yield Namespace( **dict(zip(keys, conf)) )


def get_experiment_folder(conf: Namespace, makedirs: bool=True) -> str:
    """
    Given a configuration, return path to a folder to save
    results for this configuration.

    Parameters:
        conf: Namespace - Configuration as namespace,
                          as returned by iterate_configurations
        makedirs: bool - Whether to create a folder, if not already exists
    """
    path = os.path.join(EXPERIMENT_FOLDER,
                        conf.dataset,
                        conf.feature,
                        conf.approach,
                        conf.calibration_method
                       )
    if makedirs:
        os.makedirs(path, exist_ok=True)

    return path


def gather_results(dataset: Dataset,
                   conf: Namespace
                  ) -> dict:

    data = {}

    for k in np.copy(dataset.folds):
        print(f"\nFold {k}", '~'*60)
        dataset.set_fold(k)
        dataset.select(None)

        calibrated_scores = APPROACHES[conf.approach](dataset=dataset, conf=conf)

        data[f'fold{k}'] = {
            'scores': calibrated_scores,
            'metrics': get_metrics(calibrated_scores, dataset, conf)
        }

    return data


def main():
    results_for_plotting = []

    # Try each configuration, as derived from args
    for conf in iterate_configurations():
        print("\n"+("="*80))
        print("Running on configuration", conf)

        # Save results of the experiment in this folder
        exp_folder = get_experiment_folder(conf)

        dataset = Dataset(name=conf.dataset, feature=conf.feature)

        # Check if experiment is already run
        saveto = os.path.join( exp_folder, 'results.npy' )
        if not os.path.isfile(saveto):

            # np.save(saveto, {})
            try:
                data = gather_results(dataset=dataset, conf=conf)

                with open(saveto, 'wb') as f:
                    pickle.dump(data, f)
            except Exception as e:
                data = None
                print("ERROR, could not run experiment! It gives the following error:")
                print(e)
                traceback.print_exc(limit=None, file=None, chain=True)

        else:
            with open(saveto, 'rb') as f:
                data = pickle.load(f, fix_imports=True)

        if args.visualize and data is not None:
            results_for_plotting.append((conf, data))

        dataset.select(None)
        print("\nExperiment finished, find results at", saveto)
        print(("="*80))

    if args.visualize:
        pass
        # violinplot(dataset, results_for_plotting)
        # fpr2globalfpr(dataset, results_for_plotting)

    print("Done!")


if __name__ == '__main__':
    main()
