import argparse

def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # TODO
    # parser.add_argument(
    #     "-v", "--verbose",
    #     help="Provide extra information on what process is happening.  Can be stacked up to 4 times.",
    #     action="count",
    #     default=0,
    # )

    parser.add_argument(
        '--visualize',
        help='Visualize results after running',
        action='store_true',
        default=False
    )

    parser.add_argument(
        '--dataset',
        help='name(s) of dataset',
        type=str,
        nargs="+",
        choices=['rfw', 'bfw'],
        default=['rfw', 'bfw']
    )

    parser.add_argument(
        '--feature',
        help='features',
        type=str,
        nargs="+",
        choices=['facenet', 'facenet-webface', 'arcface'],
        default=['facenet', 'facenet-webface', 'arcface']
    )

    parser.add_argument(
        '--approach',
        help='approaches',
        type=str,
        nargs='+',
        choices=['uncalibrated', 'baseline', 'oracle', 'faircal'],#, 'fsn', 'agenda', 'ftc'], TODO
        default=['uncalibrated', 'baseline', 'oracle', 'faircal'],#, 'fsn', 'agenda', 'ftc'],
    )

    parser.add_argument(
        '--calibration_method',
        help='calibration methods',
        type=str,
        nargs='+',
        choices=['beta', ],#'binning', 'isotonic_regression'], TODO
        default=['beta', ],#'binning', 'isotonic_regression'], TODO
    )

    parser.add_argument(
        '--n_cluster',
        help='Number of clusters for KMeans',
        type=int,
        nargs='+',
        default=[100,]
    )

    args = parser.parse_args()

    args = validate(args)

    return args

def validate(args: argparse.Namespace) -> argparse.Namespace:
    # TODO
    return args

args = parse()
