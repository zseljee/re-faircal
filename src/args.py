import argparse

def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--ignore_existing', '--ignore-existing',
        help='Ignore existing experiment files.  Does not ignore K-means files!',
        action='store_true',
    )

    parser.add_argument(
        '--dataset',
        help='name(s) of dataset',
        type=str,
        nargs="+",
        choices=['rfw', 'bfw'],
        default=['rfw', 'bfw'],
    )

    parser.add_argument(
        '--feature',
        help='features',
        type=str,
        nargs="+",
        choices=['facenet', 'facenet-webface', 'arcface'],
        default=['facenet', 'facenet-webface'],
    )

    parser.add_argument(
        '--approach',
        help='approaches',
        type=str,
        nargs='+',
        choices=['uncalibrated', 'baseline', 'oracle', 'faircal', 'fsn', 'ftc'],
        default=['uncalibrated', 'baseline', 'oracle', 'faircal', 'fsn'],
    )

    parser.add_argument(
        '--calibration_method', '--calibration-method',
        help='calibration methods',
        type=str,
        nargs='+',
        choices=['beta', ],
        default=['beta', ],
    )

    parser.add_argument(
        '--n_clusters', '--n-clusters',
        help='Number of clusters for K-means. ',
        type=int,
        nargs='+',
        default=[100,],
    )

    parser.add_argument(
        '--fpr_thr', '--fpr-thr',
        help='FPR used for FSN method',
        type=float,
        default=1e-3,
    )

    args = parser.parse_args()

    args = validate(args)

    return args

def validate(args: argparse.Namespace) -> argparse.Namespace:
    # TODO
    return args

args = parse()
