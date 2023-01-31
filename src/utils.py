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

    Returns:
        path: str - A path to a folder for this experiment
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
    Given a Namespace and a set of keys,
    combine the values at each key in keys with each other.
    This assumes that for each key `K`, `args.K` is a list of values,
    these values can be any dtype.

    Any unused key is returned at each yield.

    Will raise an AssertionError if a provided key does not have a list
    as its corresponding value.

    Usage:
    >>> args = Namespace(key1=['a', 'b'], key2=[1,2,3], key3='Hello world!')
    >>> keys = ['key1', 'key2']
    >>> for conf in iterate_configurations(args, keys):
    ...     print(conf)
    Namespace(key1='a', key2=1, key3='Hello world!')
    Namespace(key1='a', key2=2, key3='Hello world!')
    Namespace(key1='a', key2=3, key3='Hello world!')
    Namespace(key1='b', key2=1, key3='Hello world!')
    Namespace(key1='b', key2=2, key3='Hello world!')
    Namespace(key1='b', key2=3, key3='Hello world!')

    Parameters:
        args: Namespace
        keys: None|list[str] - A list of keys to use, if set to None, use all keys.

    Returns:
        conf: Namespace - A Namespace with the same keys as `args`

    """

    # Keys used in returned Namespace
    if keys is None:
        keys = ['dataset',
                'n_clusters',
                'feature',
                'approach',
                'calibration_method',
            ]

    # Use those keys to get a list of values for each option
    # ! IMPORTANT ! this assumes that these attributes are lists/iterables
    values = [getattr(args, key) for key in keys]

    # To stress the above point, check if `values` is a list of lists
    assert all(isinstance(val, list) for val in values), "Configuration contains non-list key"

    remainder = dict([(key, getattr(args, key)) for key in vars(args) if key not in keys])

    # Now combine each item in each list of options in values with each other
    # Ie [[1,2], ['a', 'b']] gives (1,a), (1,b), (2,a), (2,b)
    for conf in itertools.product(*values):

        # Now map the keys back to the values
        yield Namespace( **dict(zip(keys, conf)), **remainder)
