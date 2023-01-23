import itertools
from argparse import Namespace

from constants import *


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


def iterate_configurations(args: Namespace, keys: None|list[str]=None) -> Namespace:
    """
    Iterate options in args to yield configurations.
    Prevents many 'for' loops in main

    Yields Namespace with keys as defined below
    """

    # Keys used in returned Namespace
    if keys is None:
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