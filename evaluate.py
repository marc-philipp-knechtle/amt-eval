import argparse
import os
import logging
from collections import defaultdict

from datetime import datetime
from glob import glob
from typing import List

from data.dataset import SchubertWinterreiseDataset


def evaluate_inference_dir(predictions_dir: str, dataset_name: str, dataset_group: str):
    dataset_groups: List[str] = dataset_group.split(',')

    # todo replace this with a determine dataset function
    dataset = SchubertWinterreiseDataset()

    metrics = defaultdict(list)

    predictions_filepaths: List[str] = glob(os.path.join(predictions_dir, '*.mid'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('predictions_dir', type=str)
    parser.add_argument('dataset_name', nargs='?', default='default',
                        help='The dataset which the predictions are evaluated on.')
    parser.add_argument('dataset_group', nargs='?', default='',
                        help='Comma-separated dataset groups which we evaluate on.')
    parser.add_argument('--save-path', default=None)
    args: argparse.Namespace = parser.parse_args()
    dataset_name: str = parser.parse_args().dataset_name
    predictions_dir = args.predictions_dir
    """
    directory  where the predictions to be evaluated are saved
    """
    datetime_str: str = datetime.now().strftime('%y%m%d-%H%M')
    logging_filepath: str
    if args.save_path is None:
        logging_filepath = os.path.join('runs', f'evaluation-{datetime_str}.log')
    else:
        logging_filepath = os.path.join(args.save_path, f'evaluation-{dataset_name}-{datetime_str}.log')
    if not os.path.exists(os.path.dirname(logging_filepath)):
        os.makedirs(os.path.dirname(logging_filepath))
    # filemode=a -> append
    logging.basicConfig(filename=logging_filepath, filemode="a", level=logging.INFO)
    evaluate_inference_dir(predictions_dir, dataset_name, dataset_group=args.dataset_group)


if __name__ == '__main__':
    main()
