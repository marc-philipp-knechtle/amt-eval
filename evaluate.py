import argparse
import os
import logging
import re
import sys
from collections import defaultdict

from datetime import datetime
from glob import glob
from typing import List, Dict, Tuple

import mir_eval
import numpy as np
from scipy.stats import hmean
from torch import Tensor
from tqdm import tqdm

import utils.log
from constants import SAMPLE_RATE, HOP_LENGTH
from data import dataset_determination
from data.dataset import SchubertWinterreiseDataset, WagnerRingDataset, NoteTrackingDataset, ChoralSingingDataset
from data.dataset_determination import dir_contains_other_dirs
from metrics_midi import metrics_midi_nt
from utils import midi, decoding

import data

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = utils.log.CustomFormatter()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

eps = sys.float_info.epsilon

scaling_frame_to_real = HOP_LENGTH / SAMPLE_RATE
"""
With scaling_frame_to_real, we can convert from time bin indices back to realtime
"""
scaling_real_to_frame = SAMPLE_RATE / HOP_LENGTH
"""
With scaling_real_to_frame, we can convert from realtime to bin indices
"""

LOGGING_FILEPATH = ''


def determine_dataset(dataset_parameter_name: str, dataset_group: str = None) -> NoteTrackingDataset:
    dataset_groups: List[str] = dataset_group.split(',') if dataset_group else None
    dataset_class = getattr(data.dataset, dataset_parameter_name)

    kwargs = {'logger_filepath': LOGGING_FILEPATH}
    if dataset_group is not None:
        kwargs['groups'] = dataset_groups
    return dataset_class(**kwargs)


def evaluate_inference_dir(predictions_dir: str, dataset_name: str, dataset_group: str, save_path: str):
    logger.info(f'Evaluating predictions in {predictions_dir} on {dataset_name} with groups {dataset_group}.')
    dataset: NoteTrackingDataset = determine_dataset(dataset_name, dataset_group)
    metrics = evaluate_inference_dataset(dataset, predictions_dir)
    write_metrics(metrics, dataset_name, save_path)


def evaluate_inference_dataset(dataset, predictions_dir):
    metrics: defaultdict = defaultdict(list)
    predictions_filepaths: List[str] = glob(os.path.join(predictions_dir, '*.mid'))
    # path, audio, label, velocity, onset, offset, frame
    label: Tuple[str, str]
    for label in tqdm(dataset):
        audio_wav_name = os.path.basename(label[0]).replace('.wav', '')
        matching_predictions = [prediction_file for prediction_file in predictions_filepaths if
                                re.compile(fr".*{re.escape(audio_wav_name)}.*").search(prediction_file)]
        if type(dataset) is ChoralSingingDataset and len(dataset) == 5:
            matching_predictions = sorted(matching_predictions)
            matching_predictions = [matching_predictions[0]]
        if len(matching_predictions) != 1:
            raise RuntimeError(
                f'Evaluating dataset {str(dataset)}'
                f'Found different amount of predictions for label {audio_wav_name}. '
                f'Expected 1, found {len(matching_predictions)}.'
                f'length of total predictions: {len(predictions_filepaths)}')
        prediction_filepath = matching_predictions[0]
        logger.info(f'Evaluating audio wave {audio_wav_name} with prediction {prediction_filepath}.')

        file_metrics = metrics_midi_nt.calculate_metrics(prediction_filepath, label[1])
        for key, value in file_metrics.items():
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(value)
    return metrics


def write_metrics(metrics: Dict, dataset_name: str, save_path: str):
    total_eval_str: str = ''
    for key, values in metrics.items():
        if key.startswith('nt/'):
            _, category, name = key.split('/')
            eval_str: str = f'{category:>32} {name:25}: {np.mean(values, dtype=np.float64):.3f} ± {np.std(values):.3f}'
            if name == 'f1':
                """
                We compute the f1 score separately for the whole task because of issues selecting the mean. 
                Old approach -> We use the arithmetic mean of the f1 score of each prediction. 
                    The F1 score is defined as the harmonic mean of precision and recall. With this approach 
                    (using the arithmetic mean) this definition is not fulfilled.  
                """
                precision = np.mean(metrics[f'nt/{category}/precision'])
                recall = np.mean(metrics[f'nt/{category}/recall'])
                f1 = hmean([precision + eps, recall + eps]) - eps
                f1_direct_var_name: str = 'directly computed f1'
                eval_str += '\n' + f'{category:>32} {f1_direct_var_name:25}: {f1:.3f}'
            logger.info(eval_str)
            total_eval_str += eval_str + '\n'
        else:
            raise RuntimeError(f'Unknown metric key {key}.')
    if save_path is not None:
        metrics_filepath = os.path.join(save_path, f'metrics-{str(dataset_name)}.txt')
        with open(metrics_filepath, 'w') as f:
            f.write(total_eval_str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('predictions_dir', type=str)
    parser.add_argument('dataset_name', nargs='?', default='default',
                        help='The dataset which the predictions are evaluated on.')
    parser.add_argument('dataset_group', nargs='?', default=None,
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
    global LOGGING_FILEPATH
    LOGGING_FILEPATH = logging_filepath

    file_handler = logging.FileHandler(logging_filepath)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    if dir_contains_other_dirs(predictions_dir):
        all_metrics: Dict = {}
        for root, dirs, files in os.walk(predictions_dir):
            for directory in dirs:
                predictions_directory = os.path.join(root, directory)
                dataset = dataset_determination.dataset_definitions_trans_comparing_paper[directory]()
                dataset_metrics = evaluate_inference_dataset(dataset, predictions_directory)
                write_metrics(dataset_metrics, str(dataset), args.save_path)
                for key, value in dataset_metrics.items():
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].extend(value)
        write_metrics(all_metrics, 'mixed_test_set', args.save_path)
    else:
        evaluate_inference_dir(predictions_dir, dataset_name, dataset_group=args.dataset_group,
                               save_path=args.save_path)


if __name__ == '__main__':
    main()
