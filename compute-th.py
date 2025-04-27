"""


"""
import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Dict

import model_specific.models
import utils.log
from data import dataset_determination
from data.dataset import AmtEvalDataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = utils.log.CustomFormatter()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('predictions_dir', type=str)
    parser.add_argument('prediction_type', type=str)
    args: argparse.Namespace = parser.parse_args()

    if dataset_determination.dir_contains_other_dirs(args.predictions_dir):
        dataset_prediction_mapping: Dict[AmtEvalDataset, str] = {}
        for root, dirs, files in os.walk(args.predictions_dir):
            for directory in dirs:
                predictions_directory = os.path.join(root, directory)
                try:
                    dataset = dataset_determination.validation_dataset_comparing_paper[directory]()
                    dataset_prediction_mapping[dataset] = str(predictions_directory)
                except KeyError:
                    logger.warning(
                        f'Skipping dataset {directory} because it is not in validation_dataset_comparing_paper.')

        model_prediction = getattr(model_specific.models, args.prediction_type)(dataset_prediction_mapping, logger)
        logger.info(f'Computed optimal averaged threshold: {model_prediction.optimal_threshold}')


if __name__ == '__main__':
    main()
