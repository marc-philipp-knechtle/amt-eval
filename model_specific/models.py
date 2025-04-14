"""
For evaluation - two things are necessary:
1) Finding the correct prediction file for label (or predicting)
2) Calling the correct metrics for the prediction (for a MPS prediction, we need different types of metrics)

1)
The Idea of this file is to implement dynamic fetching
We can use this for multiple purpose:
* Use it to fetch an existing prediction
* Use it to call an existing model (e.g. basic-pitch) -> and create the evaluation

2)
Also the idea is to implement custom metric calculation
e.g. we want to do for O&F metric calculation on the frame output as well as on the final midis etc.
"""
import logging
import os
import re
import sys
from collections import defaultdict


import mir_eval
import numpy as np
import sklearn
import torch
from abc import abstractmethod
from glob import glob
from typing import Dict, List, Any, Tuple
from sklearn import metrics as sk_metrics

from tqdm import tqdm

import metrics_prediction.metrics_prediction_nt
import metrics.ap as ap
import utils.midi
from data.dataset import AmtEvalDataset
from metrics_midi import metrics_midi_nt
from utils import midi

from scipy.stats import hmean

eps = sys.float_info.epsilon


class ModelNTPrediction:
    dataset_prediction_mapping: Dict[AmtEvalDataset, str]
    logger: logging.Logger

    def __init__(self, dataset_prediction_mapping, logger):
        self.dataset_prediction_mapping = dataset_prediction_mapping
        self.logger = logger

    @abstractmethod
    def find_matching_midi_prediction(self, labelname, prediction_dir) -> str:
        """
        Args:
            labelname: Label to find the matching prediction for
            prediction_dir:
        Returns: the path to the prediction
        """
        return ModelNTPrediction.find_matching_file(labelname, prediction_dir, '*.mid')

    @staticmethod
    def find_matching_file(file_basename: str, directory: str, file_ending: str = None) -> str:
        """
        We separate the file_basename and file_ending to enable combinations when not the whole filename is known.
        Then, the caller can just specify a part of the filename.
        Args:
            file_basename:
            directory:
            file_ending:
        Returns: one matching filename. If there are multiple options an error is thrown.
        """
        if file_ending is not None:
            all_files = glob(os.path.join(directory, file_ending))
        else:
            all_files = glob(directory)
        matching_files = [file for file in all_files if re.compile(fr".*{re.escape(file_basename)}.*").search(file)]
        if len(matching_files) != 1:
            raise RuntimeError(
                f'Found different amount of files for file_basename {file_basename}. '
                f'Expected 1, found {len(matching_files)}.'
                f'length of total predictions: {len(directory)}')
        return matching_files[0]

    @abstractmethod
    def calculate(self, save_path: str) -> Dict:
        """
        creates all metrics possible -> returns them as dict
        """
        ...

    def write_metrics(self, metrics: Dict, dataset_name: str, save_path: str):
        total_eval_str: str = ''
        metrics = {key: val for key, val in sorted(metrics.items(), key=lambda ele: ele[0])}
        for key, values in metrics.items():
            prediction_type, category, name = key.split('/')
            """
            prediction_type = what kind of prediciton are we evaluating
            category = on what kind of level are we looking @ the category
            name = name of the metric
            """
            eval_str: str = f'{prediction_type:>32} {category:>32} {name:25}: {np.mean(values, dtype=np.float64):.3f} ± {np.std(values):.3f}'
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

                name = 'f1_from_p_r'
                eval_str += f'\n{prediction_type:>32} {category:>32} {name:25}: {f1:.3f}'
            total_eval_str += eval_str + '\n'
            self.logger.info(total_eval_str)
        if save_path is not None:
            metrics_filepath = os.path.join(save_path, f'metrics-{str(dataset_name)}.txt')
            with open(metrics_filepath, 'w') as f:
                f.write(total_eval_str)


class OnsetsAndFramesNTPrediction(ModelNTPrediction):
    """
    O&F reports by definition NT results
    -> NO MPS
    -> NO NS
    """

    OaFSampleRate: int = 16000

    SAMPLE_RATE = 16000
    HOP_LENGTH = SAMPLE_RATE * 32 // 1000  # ~ 11.2ms
    SCALING_FRAME_TO_REAL = HOP_LENGTH / SAMPLE_RATE
    SCALING_REAL_TO_FRAME = SAMPLE_RATE / HOP_LENGTH

    MIN_MIDI = 21
    MAX_MIDI = 108

    def __init__(self, dataset_prediction_mapping: Dict[AmtEvalDataset, str], logger: logging.Logger):
        super().__init__(dataset_prediction_mapping, logger)

    def __str__(self):
        return "OnsetsAndFrames"

    def calculate(self, save_path) -> Dict:
        all_metrics: Dict[str, Any] = {}
        """
        variable for all metrics for all  predictions in all datasets -> used to calculated the mixed test set
        """

        for dataset, prediction_dir in self.dataset_prediction_mapping.items():
            metrics: Dict[str, Any] = {'mpe/frame/avg_precision': [],
                                       'nt/frame/avg_precision': [],
                                       'nt/note/avg_precision': [],
                                       'nt/note-with-offset/avg_precision': []}
            compute_ap_metrics: bool = OnsetsAndFramesNTPrediction.pt_predictions_exist(prediction_dir)
            for label in tqdm(dataset):
                basename = os.path.basename(label[0]).replace('.wav', '')

                matching_midi_prediction: str = self.find_matching_midi_prediction(basename, prediction_dir)
                nt_metrics = metrics_midi_nt.calculate_metrics(matching_midi_prediction, label[1])

                if compute_ap_metrics:
                    matching_pt_prediction_frames: str = self.find_matching_pt_prediction_frames(basename,
                                                                                                 prediction_dir)
                    self.logger.info(f'Calculating frame ap for dataset: {dataset}')
                    frame_ap: float = self.calc_frame_ap(label[1], matching_pt_prediction_frames)

                    matching_pt_prediction_onsets: str = self.find_matching_pt_prediction_onsets(basename,
                                                                                                 prediction_dir)
                    self.logger.info(f'Calculating note ap for dataset: {dataset}')
                    note_ap_frame, note_ap_onset, note_ap_onset_offset = self.calc_note_ap(label[1],
                                                                                           matching_pt_prediction_frames,
                                                                                           matching_pt_prediction_onsets)

                    metrics['mpe/frame/avg_precision'].append(frame_ap)
                    metrics['nt/frame/avg_precision'].append(note_ap_frame)
                    metrics['nt/note/avg_precision'].append(note_ap_onset)
                    metrics['nt/note-with-offset/avg_precision'].append(note_ap_onset_offset)

                for key, value in nt_metrics.items():
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
            super().write_metrics(metrics, str(dataset), save_path)
            for key, value in metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].extend(value)

        super().write_metrics(all_metrics, 'mixed_test_set', save_path)
        return all_metrics

    def calc_note_ap(self, midi_path: str, frame_pt_path: str, onset_pt_path: str) -> Tuple[float, float, float]:
        """
        Calculates Average Precision for NT - like output
        Calculates Frame, Note etc. metric AFTER the onset and frame output have been combined
        We can still calculate frame metrics for this, however, they are calculated on the midi output.

        We cannot use the scipy average_precision_score, because we combine onset + frame threshold for this.
        Both must match and onset threshold must only match @ the beginning.
        Therefore, we use the raw outputs to make our own predictions ans then calculate the avg. precision manually.
        Args:
            midi_path:
            frame_pt_path:
            onset_pt_path:

        Returns:
        """
        frame_prediction: torch.FloatTensor = torch.load(frame_pt_path, map_location='cpu').cpu()
        onset_prediction: torch.FloatTensor = torch.load(onset_pt_path, map_location='cpu').cpu()
        velocities_prediction: torch.FloatTensor = torch.full_like(frame_prediction, fill_value=60, dtype=torch.float32)

        assert frame_prediction.shape == onset_prediction.shape == velocities_prediction.shape

        ref: np.ndarray = midi.parse_midi_note_tracking(midi_path)
        pitches: List = []
        intervals: List = []
        velocities: List = []
        for start_time, end_time, pitch, velocity in ref:
            pitches.append(int(pitch))
            intervals.append([start_time, end_time])
            velocities.append(velocity)
        p_ref_midi: np.ndarray = np.array(pitches)
        p_ref_hz: np.ndarray = np.array([mir_eval.util.midi_to_hz(p) for p in p_ref_midi])
        i_ref_time: np.ndarray = np.array(intervals)
        i_ref_frames: np.ndarray = (i_ref_time * self.SCALING_REAL_TO_FRAME).astype(int)
        v_ref: np.ndarray = np.array(velocities)
        # todo this might be wrong! (using frame_prediction.shape) :(
        t_ref, f_ref = self.notes_to_frames(p_ref_midi, i_ref_frames, frame_prediction.shape)
        t_ref_time = t_ref.astype(np.float64) * self.SCALING_FRAME_TO_REAL

        precision_recall_pairs_frame: List[Tuple[float, float]] = []
        precision_recall_pairs_onset: List[Tuple[float, float]] = []
        precision_recall_pairs_onset_offset: List[Tuple[float, float]] = []

        for threshold in tqdm(np.arange(0, 1.00, 0.05)):
            p_est_midi, i_est_frames, v_est = self.extract_notes(onset_prediction, frame_prediction,
                                                                 velocities_prediction, threshold, threshold)
            p_est_midi = p_est_midi + self.MIN_MIDI
            p_est_hz = np.array([mir_eval.util.midi_to_hz(p) for p in p_est_midi])
            i_est_time = (i_est_frames * self.SCALING_FRAME_TO_REAL)

            t_est, f_est = self.notes_to_frames(p_est_midi, i_est_frames, frame_prediction.shape)
            t_est_time = t_est.astype(np.float64) * self.SCALING_FRAME_TO_REAL

            frame_metrics: Dict[str, float] = mir_eval.multipitch.evaluate(t_ref_time, f_ref, t_est_time, f_est)
            precision: float = frame_metrics['Precision']
            recall: float = frame_metrics['Recall']
            precision_recall_pairs_frame.append((precision, recall))

            if len(p_est_midi) == 0:
                p_onset = 0
                r_onset = 1
                p_onset_offset = 0
                r_onset_offset = 1
            else:
                p_onset, r_onset, f, o = mir_eval.transcription.precision_recall_f1_overlap(i_ref_time, p_ref_hz,
                                                                                            i_est_time, p_est_hz,
                                                                                            offset_ratio=None)
                p_onset_offset, r_onset_offset, f, o = mir_eval.transcription.precision_recall_f1_overlap(i_ref_time,
                                                                                                          p_ref_hz,
                                                                                                          i_est_time,
                                                                                                          p_est_hz)

            precision_recall_pairs_onset.append((p_onset, r_onset))
            precision_recall_pairs_onset_offset.append((p_onset_offset, r_onset_offset))

        del p_ref_midi, p_ref_hz, i_ref_time, i_ref_frames, v_ref, t_ref, f_ref, t_ref_time
        del p_est_midi, i_est_frames, v_est, p_est_hz, i_est_time, t_est, f_est, t_est_time

        return (ap.calc_ap_from_prec_recall_pairs(precision_recall_pairs_frame),
                ap.calc_ap_from_prec_recall_pairs(precision_recall_pairs_onset),
                ap.calc_ap_from_prec_recall_pairs(precision_recall_pairs_onset_offset))

    def calc_frame_ap(self, midi_path: str, matching_prediction_pt: str) -> float:
        """
        Calculates Frame AP BEFORE the onset and frame output have been combined
        This function uses the pure frame probabilities.
        Args:
            midi_path: midi path for label to
            matching_prediction_pt:
        Returns: frame-wise average precision score
        """
        frame_prediction: torch.tensor = torch.load(matching_prediction_pt, map_location='cpu').cpu()
        columns_before = torch.zeros((frame_prediction.shape[0], 21), dtype=frame_prediction.dtype)
        columns_after = torch.zeros((frame_prediction.shape[0], 19), dtype=frame_prediction.dtype)
        frame_prediction = torch.cat((columns_before, frame_prediction), dim=1)
        frame_prediction = torch.cat((frame_prediction, columns_after), dim=1)
        """
        shape(num_of_frames, 88)
        desribes the probability that certain frame is active
        """
        note_events = utils.midi.parse_midi_note_tracking(midi_path)
        """
        shape(num_of_note_events, 4) 
        """
        f_annot_pitch = (
            metrics_prediction.metrics_prediction_nt.compute_annotation_array_nooverlap(note_events,
                                                                                        frame_prediction.shape[0],
                                                                                        self.SCALING_REAL_TO_FRAME,
                                                                                        'pitch').T)
        """
        shape(num_of_frames, 88)
        0 when note is not active at given frame and pitch, 1 otherwise
        """
        """
        # Used for debugging -> Construct midi again out of tensors (with binarization) 
        pitches, intervals, velocities = metrics_prediction.utils.extract_notes_from_frames(
            torch.tensor(f_annot_pitch), 0.5)
        intervals = intervals * self.SCALING_FRAME_TO_REAL
        midi.save_p_i_as_midi('/tmp/ref.mid', pitches, intervals, velocities)
        pitches_pred, intervals_pred, velocities_pred = metrics_prediction.utils.extract_notes_from_frames(
            frame_prediction, 0.5)
        # The pitches for the prediction are offset by 21 because this is a piano prediction -> adding 21
        pitches_pred = pitches_pred
        intervals_pred = intervals_pred * self.SCALING_FRAME_TO_REAL
        midi.save_p_i_as_midi('/tmp/pred.mid', pitches_pred, intervals_pred, velocities_pred)
        """
        avg_precision_score = sk_metrics.average_precision_score(f_annot_pitch.flatten(), frame_prediction.flatten())
        return avg_precision_score

    @staticmethod
    def pt_predictions_exist(prediction_dir) -> bool:
        return bool(glob(os.path.join(prediction_dir, '*.pt')))

    def find_matching_pt_prediction_frames(self, labelname, prediction_dir) -> str:
        return self.find_matching_prediction(labelname, prediction_dir, '*frames.pt')

    def find_matching_pt_prediction_onsets(self, labelname, prediction_dir) -> str:
        return self.find_matching_prediction(labelname, prediction_dir, '*onsets.pt')

    def find_matching_midi_prediction(self, labelname, prediction_dir) -> str:
        return self.find_matching_prediction(labelname, prediction_dir, '*.mid')

    @staticmethod
    def find_matching_prediction(labelname, prediction_dir, file_ending) -> str:
        all_predictions = glob(os.path.join(prediction_dir, file_ending))
        matching_predictions = [prediction_file for prediction_file in all_predictions if
                                re.compile(fr".*{re.escape(labelname)}\..*").search(prediction_file)]
        if len(matching_predictions) != 1:
            raise RuntimeError(
                f'Found different amount of predictions for label {labelname}. '
                f'Expected 1, found {len(matching_predictions)}.'
                f'length of total predictions: {len(prediction_dir)}')
        return matching_predictions[0]

    @staticmethod
    def extract_notes(onsets: torch.FloatTensor, frames: torch.FloatTensor, velocity: torch.FloatTensor,
                      onset_threshold=0.5, frame_threshold=0.5):
        """
        !!!
        We copied this method form the Onsets and Frames repository!
        Please check with the original implementation before modifying this code.
        !!!

        Finds the note timings based on the onsets and frames information

        Parameters
        ----------
        onsets: torch.FloatTensor, shape = [frames, bins]
        frames: torch.FloatTensor, shape = [frames, bins]
        velocity: torch.FloatTensor, shape = [frames, bins]
        onset_threshold: float
        frame_threshold: float

        Returns
        -------
        pitches:    np.ndarray of bin_indices
                    shape: (<length>, 1)
                    To my understanding, these are the pitch values for each time index
        intervals:  np.ndarray of rows containing (onset_index, offset_index)
                    shape: (<length>, 2)
                    Start and end of each note
        velocities: np.ndarray of velocity vales
                    shape: (<length>, 1)
                    Velocity value for each time index
        """
        onsets = (onsets > onset_threshold).cpu().to(torch.uint8)
        frames = (frames > frame_threshold).cpu().to(torch.uint8)
        # torch.cat = concatenates tensors. Requirement: each tensor has the same shape!
        # onsets[:1, :] = first row, keeping all columns (=time bin 0 with all possible key values)
        # onsets[1:, :] - onsets[:-1, :] = subtracts each row of onsets from the next row, creating the difference
        # This is true if the current index detects an onset and the next index does not.
        onset_diff = torch.cat([onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], dim=0) == 1
        """
        Tensor of False where there is not an onset, true wherer there is
        shape: [frames, bins] (bins=88, because of piano keys)
        """

        pitches = []
        intervals = []
        velocities = []

        for nonzero in onset_diff.nonzero():  # .nonzero() returns a tuple containing the indices of nonzero items
            frame = nonzero[0].item()
            pitch = nonzero[1].item()

            onset = frame
            offset = frame
            velocity_samples = []

            while onsets[offset, pitch].item() or frames[
                offset, pitch].item():  # as long as there is an onset which is still detected
                if onsets[offset, pitch].item():  # if there is still an onset detected
                    velocity_samples.append(velocity[offset, pitch].item())
                offset += 1
                if offset == onsets.shape[0]:  # if we reach the end of the detection
                    break

            if offset > onset:  # If we have detected sth
                pitches.append(pitch)
                intervals.append([onset, offset])
                velocities.append(np.mean(velocity_samples) if len(velocity_samples) > 0 else 0)

        return np.array(pitches), np.array(intervals), np.array(velocities)

    @staticmethod
    def notes_to_frames(pitches, intervals, shape):
        """
        !!!
        We copied this method form the Onsets and Frames repository!
        Please check with the original implementation before modifying this code.
        !!!

        Parameters
        ----------
        pitches: list of pitch bin indices
        intervals: list of [onset, offset] ranges of bin indices
        shape: the shape of the original piano roll, [n_frames, n_bins]

        Returns
        -------
        time: np.ndarray containing the frame indices
        freqs: list of np.ndarray, each containing the frequency bin indices
        """
        roll = np.zeros(tuple(shape))
        for pitch, (onset, offset) in zip(pitches, intervals):
            if pitch >= 88:
                continue
            roll[onset:offset, pitch] = 1

        time = np.arange(roll.shape[0])
        freqs = [roll[t, :].nonzero()[0] for t in time]
        return time, freqs


class BpNTPrediction(ModelNTPrediction):
    # the constant names are copied from the original repository
    # therefore it might not be possible to maintain a unified naming scheme in this repository
    FFT_HOP = 256

    SAMPLE_RATE = 22050
    AUDIO_SAMPLE_RATE = 22050

    ANNOTATIONS_FPS = AUDIO_SAMPLE_RATE // FFT_HOP
    SCALING_REAL_TO_FRAME = ANNOTATIONS_FPS
    SCALING_FRAME_TO_REAL = 1.0 / ANNOTATIONS_FPS

    def __init__(self, dataset_prediction_mapping: Dict[AmtEvalDataset, str], logger: logging.Logger):
        super().__init__(dataset_prediction_mapping, logger)

    def __str__(self):
        return "BasicPitch"

    def find_matching_midi_prediction(self, labelname, prediction_dir) -> str:
        return super().find_matching_midi_prediction(labelname, prediction_dir)

    def calculate(self, save_path: str) -> Dict:
        all_metrics: Dict[str, Any] = {}
        dataset: AmtEvalDataset
        prediction_dir: str
        for dataset, prediction_dir in self.dataset_prediction_mapping.items():
            metrics: Dict[str, Any] = {'mpe/frame/avg_precision': [],
                                       'nt/frame/avg_precision': [],
                                       'nt/note/avg_precision': [],
                                       'nt/note-with-offset/avg_precision': []}
            compute_ap_metrics: bool = True
            for label in tqdm(dataset):
                basename: str = str(os.path.basename(label[0]).replace('.wav', ''))
                matching_midi_prediction: str = self.find_matching_midi_prediction(basename, prediction_dir)

                nt_metrics = metrics_midi_nt.calculate_metrics(matching_midi_prediction, label[1])

                if compute_ap_metrics:
                    matching_npz_file: str = super().find_matching_file(basename, prediction_dir, '*.npz')
                    data: np.ndarray = np.load(matching_npz_file, allow_pickle=True)['basic_pitch_model_output'].item()

                    note: np.ndarray = data['note']
                    contour: np.ndarray = data['contour']
                    onset: np.ndarray = data['onset']

                    self.logger.info(f'Calculating frame ap for dataset: {dataset}')

                    self.calc_frame_ap(matching_midi_prediction, note)

                    # todo calculate AP values

        return all_metrics

    def calc_frame_ap(self, midi_path: str, note: np.ndarray):
        note_events = utils.midi.parse_midi_note_tracking(midi_path)
        columns_before = np.zeros((note.shape[0], 21), dtype=note.dtype)
        columns_after = np.zeros((note.shape[0], 19), dtype=note.dtype)
        note = np.concatenate((columns_before, note), axis=1)
        note = np.concatenate((note, columns_after), axis=1)
        """
        shape(num_of_note_events, 4)
        """
        f_annot_pitch = (metrics_prediction.metrics_prediction_nt.compute_annotation_array_nooverlap(note_events,
                                                                                                     note.shape[0],
                                                                                                     self.SCALING_REAL_TO_FRAME,
                                                                                                     'pitch').T)
        print('asdf')
