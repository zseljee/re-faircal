import os
from typing import Any, Iterator
from argparse import Namespace
import itertools

import numpy as np

from args import args
from constants import *
from dataset import Dataset

from faircal import faircal

APPROACHES = {
    'baseline': NotImplementedError(),
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


def get_experiment_folder(conf: Namespace, makedirs: bool=False) -> str:
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

    for fold in dataset.folds:
        # Set to train fold
        dataset.select(fold=[f for f in dataset.folds if f != fold])

        data[f'fold{fold}'] = APPROACHES[conf.approach](dataset=dataset, conf=conf)
    
    return data




def main():
    # Try each configuration, as derived from args
    for conf in iterate_configurations():
        print("\n"+("="*80))
        print("Running on configuration", conf)

        # Save results of the experiment in this folder
        exp_folder = get_experiment_folder(conf)

        dataset = Dataset(dataset=conf.dataset, feature=conf.feature)
        
        # Check if experiment is already run
        saveto = os.path.join( exp_folder, 'results.npy' )
        print("Saving results to", saveto)
        if os.path.isfile(saveto):
            print("Results file already exists, skipping")
            continue
        
        # np.save(saveto, {})
        data = gather_results(dataset=dataset, conf=conf)
        print(data)
        # np.save(saveto, data)
        

if __name__ == '__main__':
    main()