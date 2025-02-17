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


def evaluate_inference_dir(predictions_dir: str, dataset_name: str, dataset_group: str, save_path: str = None):
    logger.info(f'Evaluating predictions in {predictions_dir} on {dataset_name} with groups {dataset_group}. '
                f'Storing results in {save_path}.')

    dataset: NoteTrackingDataset = determine_dataset(dataset_name, dataset_group)

    evaluate_inference_dataset(dataset, predictions_dir, save_path)


def evaluate_inference_dataset(dataset, predictions_dir, save_path):
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
        prediction_note_tracking: np.ndarray = midi.parse_midi_note_tracking(prediction_filepath)

        pitches: List[int] = []
        intervals: List[Tuple[float, float]] = []
        velocities: List[int] = []
        for start_time, end_time, pitch, velocity in prediction_note_tracking:
            if start_time == end_time:
                continue

            pitches.append(int(pitch))
            intervals.append((float(start_time), float(end_time)))
            velocities.append(velocity)

        p_est: np.ndarray = np.array(pitches)
        """
        Array of estimated pitches (in midi values)
        shape=(n,)
        """
        p_est_hz: np.ndarray = np.array([mir_eval.util.midi_to_hz(p) for p in p_est])
        """
        shape=(n,)
        """
        i_est: np.ndarray = np.array(intervals).reshape(-1, 2)
        """
        List of estimated intervals (onset time, offset time), in real! time
        shape=(n,2)
        """
        v_est: np.ndarray = np.array(velocities)
        """
        shape=(n,)
        """
        del pitches, intervals, velocities, prediction_note_tracking

        label_note_tracking: np.ndarray = midi.parse_midi_note_tracking(label[1])
        pitches_ref: List[int] = []
        intervals_ref: List[Tuple[float, float]] = []
        velocities_ref: List[int] = []
        for start_time, end_time, pitch, velocity in label_note_tracking:
            pitches_ref.append(int(pitch))
            intervals_ref.append((float(start_time), float(end_time)))
            velocities_ref.append(velocity)
        p_ref: np.ndarray = np.array(pitches_ref)
        """
        shape=(m,)
        Array of reference pitches (in midi values)
        """
        p_ref_hz: np.ndarray = np.array([mir_eval.util.midi_to_hz(p) for p in p_ref])
        i_ref: np.ndarray = np.array(intervals_ref).reshape(-1, 2)
        """
        shape=(m,2)
        """
        v_ref: np.ndarray = np.array(velocities_ref)
        """
        shape=(m,)
        """
        del pitches_ref, intervals_ref, velocities_ref

        p, r, f, o = mir_eval.transcription.precision_recall_f1_overlap(i_ref, p_ref_hz,
                                                                        i_est, p_est_hz,
                                                                        offset_ratio=None)
        logger.debug(f"Calculated onset metrics p: {p}, r: {r}, f: {f}, o: {o}")
        metrics['metric/note/precision'].append(p)
        metrics['metric/note/recall'].append(r)
        metrics['metric/note/f1'].append(f)
        metrics['metric/note/overlap'].append(o)

        p, r, f, o = mir_eval.transcription.precision_recall_f1_overlap(i_ref, p_ref_hz, i_est, p_est_hz)
        logger.debug(f"Calculated onset/offset metrics p: {p}, r: {r}, f: {f}, o: {o}")
        metrics['metric/note-with-offsets/precision'].append(p)
        metrics['metric/note-with-offsets/recall'].append(r)
        metrics['metric/note-with-offsets/f1'].append(f)
        metrics['metric/note-with-offsets/overlap'].append(o)

        p, r, f, o = mir_eval.transcription_velocity.precision_recall_f1_overlap(i_ref, p_ref_hz, v_ref,
                                                                                 i_est, p_est_hz, v_est,
                                                                                 offset_ratio=None,
                                                                                 velocity_tolerance=0.1)
        logger.debug(f"Calculated onset/velocity metrics p: {p}, r: {r}, f: {f}, o: {o}")
        metrics['metric/note-with-velocity/precision'].append(p)
        metrics['metric/note-with-velocity/recall'].append(r)
        metrics['metric/note-with-velocity/f1'].append(f)
        metrics['metric/note-with-velocity/overlap'].append(o)

        p, r, f, o = mir_eval.transcription_velocity.precision_recall_f1_overlap(i_ref, p_ref_hz, v_ref,
                                                                                 i_est, p_est_hz, v_est,
                                                                                 velocity_tolerance=0.1)
        logger.debug(f"Calculated onset/offset/velocity metrics p: {p}, r: {r}, f: {f}, o: {o}")
        metrics['metric/note-with-offsets-and-velocity/precision'].append(p)
        metrics['metric/note-with-offsets-and-velocity/recall'].append(r)
        metrics['metric/note-with-offsets-and-velocity/f1'].append(f)
        metrics['metric/note-with-offsets-and-velocity/overlap'].append(o)

        frame_metrics = evaluate_note_based_mpe(p_ref, i_ref, p_est, i_est)
        for key, loss in frame_metrics.items():
            metrics['metric/frame/' + key.lower().replace(' ', '_')].append(loss)
        metrics['metric/frame/f1'].append(
            hmean([frame_metrics['Precision'] + eps, frame_metrics['Recall'] + eps]) - eps)

        del i_ref, i_est, p_est, p_ref, p_est_hz, p_ref_hz
    total_eval_str: str = ''
    for key, values in metrics.items():
        if key.startswith('metric/'):
            _, category, name = key.split('/')
            eval_str: str = f'{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}'
            logger.info(eval_str)
            total_eval_str += eval_str + '\n'
    if save_path is not None:
        metrics_filepath = os.path.join(save_path, f'metrics-{str(dataset)}.txt')
        with open(metrics_filepath, 'w') as f:
            f.write(total_eval_str)


def evaluate_note_based_mpe(p_ref: np.ndarray, i_ref: np.ndarray, p_est: np.ndarray, i_est: np.ndarray):
    """
    :param p_ref: pitch values, shape(n,)
    :param i_ref: reference intervals, shape(n,2)
    :param p_est: estimated pitch values, shape(m,1) (m=number of detected notes)
    :param i_est: estimated intervals, shape(m,2)
    """
    i_est_frames: np.ndarray = (i_est * scaling_real_to_frame).astype(int).reshape(-1, 2)
    i_ref_frames: np.ndarray = (i_ref * scaling_real_to_frame).astype(int).reshape(-1, 2)

    p_ref_min_midi = np.array([x for x in p_ref])
    t_ref, f_ref = decoding.note_to_multipitch_realtime(p_ref_min_midi, i_ref_frames, scaling_frame_to_real)
    """
    List of estimated intervals in frame time, length n is the number of estimated notes
    shape=(m,2)
    """
    t_est, f_est = decoding.note_to_multipitch_realtime(p_est, i_est_frames, scaling_frame_to_real)

    return mir_eval.multipitch.evaluate(t_ref, f_ref, t_est, f_est)


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
        for root, dirs, files in os.walk(predictions_dir):
            for directory in dirs:
                local_dir_path = os.path.join(root, directory)
                dataset = dataset_determination.dataset_definitions_trans_comparing_paper[directory]()
                evaluate_inference_dataset(dataset, local_dir_path, args.save_path)
    else:
        evaluate_inference_dir(predictions_dir, dataset_name, dataset_group=args.dataset_group,
                               save_path=args.save_path)


if __name__ == '__main__':
    main()
