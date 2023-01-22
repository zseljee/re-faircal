import os
import pickle
from typing import Any, Iterator
from argparse import Namespace
import itertools

import numpy as np

from args import args
from constants import *
from dataset import Dataset

from approaches import faircal, baseline
from visualisations import violinplot

APPROACHES = {
    'baseline': baseline,
    'faircal': faircal
}

def iterate_configurations() -> Namespace:
    """
    Iterate options in args to yield configurations.
    Prevents many 'for' loops in main

    Yields Namespace with keys as defined below
    """

    # Keys used in returned Namespace
    # ! Important! The order of 'keys' has to match with the order of 'values'
    keys = ['dataset',
            'n_cluster',
            'fpr_thr',
            'feature',
            'approach',
            'calibration_method',
           ]

    # List of lists of each value to try
    # Ie values = [ ['A','B'], [1,2,3] ] will give A1, A2, A3, B1, B2, B3
    values = [args.datasets,
              args.n_clusters,
              args.fpr_thrs,
              args.features,
              args.approaches,
              args.calibration_methods,
             ]
    
    # Make sure we can map each value in each of the lists with a key
    assert len(keys) == len(values), "Cannot combine these names with these values!"

    # Combine each item in each list in values with all other items
    for conf in itertools.product(*values):

        # Use the list of keys defined above to create a namespace
        yield Namespace(
            **dict(zip(keys, conf))
        )


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

        data[f'fold{k}'] = APPROACHES[conf.approach](dataset=dataset, conf=conf)
    
    return data


def main():
    results_for_plotting = dict()

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
            data = gather_results(dataset=dataset, conf=conf)

            with open(saveto, 'wb') as f:
                pickle.dump(data, f)
        
        else:
            with open(saveto, 'rb') as f:
                data = pickle.load(f, fix_imports=True)

        if args.visualize:
            results_for_plotting[conf.approach] = data

        dataset.select(None)
        print("\nExperiment finished, find results at", saveto)
        print(("="*80))
    
    if args.visualize:
        violinplot(**results_for_plotting)

    print("Done!")
        

if __name__ == '__main__':
    main()