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
from abc import abstractmethod
from glob import glob
from typing import Dict, List, Any, Tuple

import basic_pitch.note_creation
import mir_eval
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import hmean
from sklearn import metrics as sk_metrics
from sklearn.metrics import PrecisionRecallDisplay
from tqdm import tqdm

import metrics.ap as ap
import metrics_prediction.metrics_prediction_nt
import utils.midi
import visualizations.plots
from data.dataset import AmtEvalDataset
from metrics_midi import metrics_midi_nt
from utils import midi

eps = sys.float_info.epsilon


class ModelNTPrediction:
    dataset_prediction_mapping: Dict[AmtEvalDataset, str]
    logger: logging.Logger

    OPTIMAL_THR_FILEENDING = 'opt-thr'

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
        return self.find_matching_file(labelname, prediction_dir, '*.mid')

    def find_matching_file(self, file_basename: str, directory: str, file_ending: str = None) -> str:
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
        matching_files = [file for file in matching_files if self.OPTIMAL_THR_FILEENDING not in file]
        if len(matching_files) != 1:
            # todo fix this edge case with real handling for CSD files
            if len(matching_files) == 5:
                # edge case handling the CSD naming scheme
                """
                basename: CSD_Guerrerro_NinoDios
                (without all the nobas etc. stuff) 
                ['/media/mpk/external-nvme/predictions/bp-04-24-comp+maestro/comparing/CSD/CSD_Guerrero_NinoDios_noten_basic_pitch.npz', '/media/mpk/external-nvme/predictions/bp-04-24-comp+maestro/comparing/CSD/CSD_Guerrero_NinoDios_basic_pitch.npz', '/media/mpk/external-nvme/predictions/bp-04-24-comp+maestro/comparing/CSD/CSD_Guerrero_NinoDios_nosop_basic_pitch.npz', '/media/mpk/external-nvme/predictions/bp-04-24-comp+maestro/comparing/CSD/CSD_Guerrero_NinoDios_noalt_basic_pitch.npz', '/media/mpk/external-nvme/predictions/bp-04-24-comp+maestro/comparing/CSD/CSD_Guerrero_NinoDios_nobas_basic_pitch.npz']
                """
                matching_files = sorted(matching_files)
                return matching_files[0]
            raise RuntimeError(
                f'Found different amount of files for file_basename {file_basename}. '
                f'Expected 1, found {len(matching_files)}.'
                f'length of total predictions: {len(directory)}')
        return matching_files[0]

    def get_p_i_v_attributes_from_midi(self, midi_path: str, shape, scaling_real_to_frame, scaling_frame_to_real):
        p_midi, i_time, v = midi.get_p_i_v_from_midi(midi_path)
        i_frames = (i_time * scaling_real_to_frame).astype(int)
        p_hz = np.array([mir_eval.util.midi_to_hz(p) for p in p_midi])

        # todo this might be wrong! (giving the shape directly) :(
        # todo this might be wrong because of way notes_to_frames is implemented -> it stops after 88 piano keys
        t, f = self.notes_to_frames(p_midi, i_frames, shape)
        t_time = t.astype(np.float64) * scaling_frame_to_real

        return {
            'p_midi': p_midi,
            'p_hz': p_hz,
            'i_time': i_time,
            'i_frames': i_frames,
            'v': v,
            't': t,
            't_time': t_time,
            'f': f
        }

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
        time: np.ndarray containing the frame indices, these should be evenly spaced
        freqs: list of np.ndarray, each containing the frequency bin indices. The freqs for a frame indice can be empty
        """
        roll = np.zeros(tuple(shape))
        for pitch, (onset, offset) in zip(pitches, intervals):
            if pitch >= 88:
                continue
            roll[onset:offset, pitch] = 1

        time = np.arange(roll.shape[0])
        freqs = [roll[t, :].nonzero()[0] for t in time]
        return time, freqs

    @abstractmethod
    def calculate(self, save_path: str, **kwargs) -> Dict:
        """
        creates all metrics possible -> returns them as dict
        """
        ...

    def write_metrics(self, metrics: Dict, dataset_name: str, save_path: str):
        prediction_type = 'Type of prediction (e.g. frame output, nt output)'
        category = 'Evaluation Target (e.g. looking @notes)'
        name = 'type of metric'
        total_eval_str: str = f'{prediction_type:>32} {category:>32} {name:25}:\n'
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
                precision = np.mean(metrics[f'{prediction_type}/{category}/precision'])
                recall = np.mean(metrics[f'{prediction_type}/{category}/recall'])
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
    HOP_LENGTH = SAMPLE_RATE * 32 // 1000  # ~ 11.2ms (32 -> see O+F repo = length of a single frame in ms)
    SCALING_FRAME_TO_REAL = HOP_LENGTH / SAMPLE_RATE
    SCALING_REAL_TO_FRAME = SAMPLE_RATE / HOP_LENGTH

    MIN_MIDI = 21
    MAX_MIDI = 108

    DEAFUL_ONSET_THRESHOLD = 0.5
    DEFAULT_FRAME_THRESHOLD = 0.5

    def __init__(self, dataset_prediction_mapping: Dict[AmtEvalDataset, str], logger: logging.Logger):
        super().__init__(dataset_prediction_mapping, logger)

    def __str__(self):
        return "OnsetsAndFramesNTPrediction"

    def optimal_threshold(self) -> float:
        """
        Inspired by:
        https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
        Returns: the optimal threshold for the dataset, combined in onset & frame threshold!
        """
        best_thr_foreach_file: List[float] = []
        onset_values_for_diagram = {np.round(x, decimals=2): [] for x in np.arange(0.05, 0.9, 0.05)}
        frame_values_for_diagram = {np.round(x, decimals=2): [] for x in np.arange(0.05, 0.9, 0.05)}

        for dataset, prediction_dir in self.dataset_prediction_mapping.items():
            assert OnsetsAndFramesNTPrediction.pt_predictions_exist(prediction_dir)
            for label in tqdm(dataset):
                basename = os.path.basename(label[0]).replace('.wav', '')
                matching_pt_prediction_frames: str = self.find_matching_pt_prediction_frames(basename, prediction_dir)
                frame_prediction: torch.FloatTensor = torch.load(matching_pt_prediction_frames,
                                                                 map_location='cpu').cpu()
                try:
                    matching_pt_prediction_onsets: str = self.find_matching_pt_prediction_onsets(basename,
                                                                                                 prediction_dir)
                    velocities_prediction = (
                        torch.full_like(frame_prediction, fill_value=60, dtype=torch.float32))  # noqa
                    onset_prediction = torch.load(matching_pt_prediction_onsets, map_location='cpu').cpu()
                except RuntimeError:
                    onset_prediction = None
                    velocities_prediction = None

                ref: dict = self.get_p_i_v_attributes_from_midi(label[1], frame_prediction.shape,
                                                                self.SCALING_REAL_TO_FRAME, self.SCALING_FRAME_TO_REAL)

                best_threshold: float = -1
                best_threshold_value: np.float64 = np.float64(-1)

                for threshold in np.arange(0.05, 0.9, 0.05):
                    threshold = np.round(threshold, decimals=2)
                    # -> Using F1 score (harmonic mean) between precision and recall as important value
                    est: dict = self.get_p_i_v_from_tensor(frame_prediction, onset_prediction, velocities_prediction,
                                                           threshold, threshold)
                    frame_metrics: Dict[str, float] = mir_eval.multipitch.evaluate(ref['t_time'], ref['f'],
                                                                                   est['t_time'], est['f'])
                    frame_f1 = hmean([frame_metrics['Precision'] + eps, frame_metrics['Recall'] + eps]) - eps

                    if len(est['i_time']) == 0:
                        f_onset = 0.0
                    else:
                        p_onset, r_onset, f_onset, o_onset = mir_eval.transcription.precision_recall_f1_overlap(
                            ref['i_time'],
                            ref['p_hz'],
                            est['i_time'],
                            est['p_hz'], offset_ratio=None)

                    onset_values_for_diagram[threshold].append(f_onset)
                    frame_values_for_diagram[threshold].append(frame_f1)

                    current_val: np.float64 = np.mean([frame_f1, f_onset])  # noqa (returns float64 not ndarray)
                    if best_threshold_value < current_val:
                        best_threshold_value = current_val
                        best_threshold = threshold

                best_thr_foreach_file.append(best_threshold)

                opt_thr_values = []
                for thr, values in onset_values_for_diagram.items():
                    opt_thr_values.append(np.mean(values))

        # visualizations.plots.plot_threshold_optimization(onset_values_for_diagram, frame_values_for_diagram)

        return float(np.mean(best_thr_foreach_file))

    def calculate(self, save_path, **kwargs) -> Dict:
        all_metrics: Dict[str, Any] = {}
        """
        variable for all metrics for all  predictions in all datasets -> used to calculated the mixed test set
        """
        frame_threshold = kwargs.get('frame_threshold')
        onset_threshold = kwargs.get('onset_threshold')
        calc_ap: bool = kwargs.get('calc_ap', False)

        for dataset, prediction_dir in self.dataset_prediction_mapping.items():
            metrics = {}
            compute_ap_metrics: bool = OnsetsAndFramesNTPrediction.pt_predictions_exist(prediction_dir) and calc_ap
            for label in tqdm(dataset):
                basename = os.path.basename(label[0]).replace('.wav', '')

                ap_metrics = {}
                nt_metrics = {}
                mpe_metrics = {}
                if compute_ap_metrics:
                    ap_metrics = self.calc_ap_values(basename, prediction_dir, label[1])

                # if we just calculate the metrics by using the midi file
                if frame_threshold is None and onset_threshold is None:
                    matching_midi_prediction: str = self.find_matching_midi_prediction(basename, prediction_dir)
                    nt_metrics: Dict[str, float] = metrics_midi_nt.calculate_metrics(matching_midi_prediction, label[1])
                    mpe_metrics = self.calc_metric_and_save_midi_from_frames(basename, prediction_dir,
                                                                             self.DEAFUL_ONSET_THRESHOLD,
                                                                             self.DEFAULT_FRAME_THRESHOLD, label[1])
                # if we calculate the metrics by using the raw predictions + extract the midi out of this prediction
                elif frame_threshold is not None and onset_threshold is not None:
                    # todo uncomment this for pure frame level eval
                    nt_metrics: Dict[str, float] = self.calc_metric_and_save_midi(basename, prediction_dir,
                                                                                  onset_threshold, frame_threshold,
                                                                                  label[1])
                    nt_metrics_onset = self.calc_onset_metrics(label[1],
                                                               self.find_matching_pt_prediction_onsets(basename,
                                                                                                       prediction_dir),
                                                               onset_threshold)
                    mpe_metrics = self.calc_metric_and_save_midi_from_frames(basename, prediction_dir, onset_threshold,
                                                                             frame_threshold, label[1])
                else:
                    raise RuntimeError(
                        f'You need to set frame_threshold as well as onset_threshold. You need to set both.')

                joined_metrics: Dict[str, float] = {**nt_metrics, **ap_metrics, **mpe_metrics, **nt_metrics_onset}
                for key, value in joined_metrics.items():
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

    def calc_metric_and_save_midi(self, basename, prediction_dir: str, onset_threshold: float,
                                  frame_threshold: float, ref_midi_path: str) -> Dict[str, float]:
        pt_pred_frames: str = self.find_matching_pt_prediction_frames(basename, prediction_dir)
        pt_pred_onsets: str = self.find_matching_pt_prediction_onsets(basename, prediction_dir)
        frames: torch.FloatTensor = torch.load(pt_pred_frames, map_location='cpu').cpu()
        onsets: torch.FloatTensor = torch.load(pt_pred_onsets, map_location='cpu').cpu()
        velocities: torch.FloatTensor = torch.full_like(frames, fill_value=60, dtype=torch.float32)  # noqa

        est = self.get_p_i_v_from_tensor(frames, onsets, velocities, onset_threshold, frame_threshold)

        threshold_optimized_midi_filepath = os.path.join(prediction_dir, f'{basename}-opt-thr.mid')
        midi.save_p_i_as_midi(threshold_optimized_midi_filepath, est['p_midi'], est['i_time'], est['v'])

        return metrics_midi_nt.calculate_metrics(threshold_optimized_midi_filepath, ref_midi_path)

    def calc_metric_and_save_midi_from_frames(self, basename, prediction_dir, onset_threshold, frame_threshold,
                                              ref_midi_path) -> Dict[str, float]:
        pt_pred_frames: str = self.find_matching_pt_prediction_frames(basename, prediction_dir)
        frames: torch.FloatTensor = torch.load(pt_pred_frames, map_location='cpu').cpu()

        est = self.get_p_i_v_from_tensor(frames, onset_threshold=onset_threshold, frame_threshold=frame_threshold)
        midi_filepath = os.path.join(prediction_dir, f'{basename}-opt-thr-fromframes.mid')
        midi.save_p_i_as_midi(midi_filepath, est['p_midi'], est['i_time'], est['v'])

        return metrics_midi_nt.calculate_metrics(midi_filepath, ref_midi_path, pred_source='mpe')

    def calc_ap_values(self, basename, prediction_dir, midipath: str) -> Dict[str, float]:
        self.logger.info(f'Calculating ap values for label: {basename}')
        matching_pt_prediction_frames: str = self.find_matching_pt_prediction_frames(basename, prediction_dir)
        # todo comment for frame model
        matching_pt_prediction_onsets: str = self.find_matching_pt_prediction_onsets(basename, prediction_dir)

        return {
            'mpe/frame-raw/avg_precision': self.calc_mpe_frame_ap(midipath, matching_pt_prediction_frames),
            'mpe/frame/avg_precision': self.calc_frame_ap(midipath, matching_pt_prediction_frames),
            # todo comment for frame model
            'nt/frame/avg_precision': self.calc_note_ap(midipath, matching_pt_prediction_frames,
                                                        matching_pt_prediction_onsets),
            # todo comment for frame model
            'nt/onset-raw/avg_precision': self.calc_onset_ap(midipath, matching_pt_prediction_onsets)
        }

    def calc_frame_ap(self, midi_path: str, frame_pt_path: str):
        frame_prediction: torch.FloatTensor = torch.load(frame_pt_path, map_location='cpu').cpu()
        # todo this might be wrong! (using frame_prediction.shape) :(
        ref = self.get_p_i_v_attributes_from_midi(midi_path, frame_prediction.shape, self.SCALING_REAL_TO_FRAME,
                                                  self.SCALING_FRAME_TO_REAL)
        precision_recall_pairs_frame: List[Tuple[float, float]] = []

        executed_thresholds = []  # Just used for visualization purposes
        # includes threshold 0, but excludes threshold 1.0 -> 1.0 = no predictions -> not useful...
        for threshold in np.arange(0, 1.0, 0.05):
            est = self.get_p_i_v_from_tensor(frame_prediction, frame_threshold=threshold)

            if len(est['p_midi']) == 0:
                continue

            frame_metrics: Dict[str, float] = mir_eval.multipitch.evaluate(ref['t_time'], ref['f'], est['t_time'],
                                                                           est['f'])
            p_frame: float = frame_metrics['Precision']
            r_frame: float = frame_metrics['Recall']
            precision_recall_pairs_frame.append((p_frame, r_frame))

            executed_thresholds.append(threshold)
            precision_recall_pairs_frame.append((p_frame, r_frame))

        thresholds = np.arange(0, 1.0, 0.05).tolist()
        return ap.calc_ap_from_prec_recall_pairs(precision_recall_pairs_frame, plot=False, thresholds=thresholds)

    def get_p_i_v_from_tensor(self, frame_prediction: torch.FloatTensor, onset_prediction: torch.FloatTensor = None,
                              velocities_prediction: torch.FloatTensor = None,
                              onset_threshold=0.5, frame_threshold=0.5) -> Dict[str, np.ndarray | List]:
        """
        Takes raw model output and returns Dict where the values can be used in mir_eval
        onset_prediction and velocities prediction is optional -> if not specified, everything is computed from frames
        Args:
            frame_prediction: frame model output
            onset_prediction: onset model output
            velocities_prediction: velocities model output
            onset_threshold:
            frame_threshold:
        Returns: Dict
        """
        if onset_prediction is not None and velocities_prediction is not None:
            p_midi, i_frames, v = self.extract_notes(onset_prediction, frame_prediction, velocities_prediction,
                                                     onset_threshold, frame_threshold)
        elif frame_prediction is not None and onset_prediction is None and velocities_prediction is None:
            p_midi, i_frames, v = self.extract_notes_from_frames(frame_prediction, frame_threshold)
        else:
            raise RuntimeError('Unknown combination of inputs')
        p_midi = p_midi + self.MIN_MIDI
        p_hz = np.array([mir_eval.util.midi_to_hz(p) for p in p_midi])
        i_time = (i_frames * self.SCALING_FRAME_TO_REAL)

        t, f = self.notes_to_frames(p_midi, i_frames, frame_prediction.shape)
        t_time = t.astype(np.float64) * self.SCALING_FRAME_TO_REAL
        return {
            'p_midi': p_midi,
            'p_hz': p_hz,
            'i_time': i_time,
            'i_frames': i_frames,
            'v': v,
            't': t,
            't_time': t_time,
            'f': f
        }

    def calc_note_ap(self, midi_path: str, frame_pt_path: str, onset_pt_path: str) -> float:
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
        velocities: torch.FloatTensor = torch.full_like(frame_prediction, fill_value=60, dtype=torch.float32)  # noqa

        assert frame_prediction.shape == onset_prediction.shape == velocities.shape

        # todo this might be wrong! (using frame_prediction.shape) :(
        ref = self.get_p_i_v_attributes_from_midi(midi_path, frame_prediction.shape, self.SCALING_REAL_TO_FRAME,
                                                  self.SCALING_FRAME_TO_REAL)

        precision_recall_pairs_frame: List[Tuple[float, float]] = []
        thresholds = []
        # includes threshold 0, but excludes threshold 1.0 -> 1.0 = no predictions -> not useful...
        for threshold in np.arange(0, 1.0, 0.05):
            est = self.get_p_i_v_from_tensor(frame_prediction, onset_prediction, velocities,
                                             threshold, threshold)
            if len(est['p_midi']) == 0:
                continue
            frame_metrics: Dict[str, float] = mir_eval.multipitch.evaluate(ref['t_time'], ref['f'], est['t_time'],
                                                                           est['f'])
            p_frame: float = frame_metrics['Precision']
            r_frame: float = frame_metrics['Recall']

            thresholds.append(threshold)
            precision_recall_pairs_frame.append((p_frame, r_frame))

        return ap.calc_ap_from_prec_recall_pairs(precision_recall_pairs_frame, plot=False, thresholds=thresholds)

    def calc_mpe_frame_ap(self, midi_path: str, matching_prediction_pt: str) -> float:
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
        avg_precision_score = sk_metrics.average_precision_score(f_annot_pitch.flatten(), frame_prediction.flatten())
        return avg_precision_score

    def calc_onset_ap(self, midi_path: str, matching_onset_prediction_pt_path: str, plot=False) -> float:
        onset_prediction: torch.tensor = torch.load(matching_onset_prediction_pt_path, map_location='cpu').cpu()

        # Extending with zeros to match the 88 piano keys (extend to full piano range)
        columns_before = torch.zeros((onset_prediction.shape[0], 21), dtype=onset_prediction.dtype)
        columns_after = torch.zeros((onset_prediction.shape[0], 19), dtype=onset_prediction.dtype)
        onset_prediction = torch.cat((columns_before, onset_prediction), dim=1)
        onset_prediction = torch.cat((onset_prediction, columns_after), dim=1)

        note_events = utils.midi.parse_midi_note_tracking(midi_path)
        f_annot_onsets = (
            metrics_prediction.metrics_prediction_nt.compute_onset_array_nooverlap(note_events,
                                                                                   onset_prediction.shape[0],
                                                                                   self.SCALING_REAL_TO_FRAME,
                                                                                   'pitch').T)

        avg_precision_score = sk_metrics.average_precision_score(f_annot_onsets.flatten(), onset_prediction.flatten())
        if plot:
            PrecisionRecallDisplay.from_predictions(f_annot_onsets.flatten(), onset_prediction.flatten())
            plt.show()
        return avg_precision_score

    def calc_onset_metrics(self, midi_path: str, matching_onset_prediction_pt_path: str, onset_threshold: float) -> \
            Dict[str, float]:
        onset_prediction: torch.tensor = torch.load(matching_onset_prediction_pt_path, map_location='cpu').cpu()
        onsets = (onset_prediction > onset_threshold).cpu().to(torch.uint8)

        columns_before = torch.zeros((onset_prediction.shape[0], 21), dtype=onset_prediction.dtype)
        columns_after = torch.zeros((onset_prediction.shape[0], 19), dtype=onset_prediction.dtype)
        onsets = torch.cat((columns_before, onsets), dim=1)
        onsets = torch.cat((onsets, columns_after), dim=1)

        note_events = utils.midi.parse_midi_note_tracking(midi_path)
        f_annot_onsets = (
            metrics_prediction.metrics_prediction_nt.compute_onset_array_nooverlap(note_events,
                                                                                   onset_prediction.shape[0],
                                                                                   self.SCALING_REAL_TO_FRAME,
                                                                                   'pitch').T)

        precision = sk_metrics.precision_score(f_annot_onsets.flatten(), onsets.flatten())
        recall = sk_metrics.recall_score(f_annot_onsets.flatten(), onsets.flatten())
        f1 = sk_metrics.f1_score(f_annot_onsets.flatten(), onsets.flatten())
        return {
            'nt/onset-raw/precision': precision,
            'nt/onset-raw/recall': recall,
            'nt/onset-raw/f1': f1
        }

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
                      onset_threshold: float = 0.5, frame_threshold: float = 0.5):
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
    def extract_notes_from_frames(frames, threshold):
        """
        With this, you can use just the frame output to extract the notes
        This does not use the onsets.
        We just use a custom threshold to specify after what number of notes we consider a note to be detected.
        Returns:
            pitches:           np array of detected pitches (in midi)
            intervals:         np array of detected intervals (in realtime)
            velocities:        np array of detected velocities
        """
        # sets each value to 1 if it is above the threshold, 0 otherwise
        frames = (frames > threshold).cpu().to(torch.uint8)  # noqa
        pitches = []
        intervals = []
        velocities = []

        onsets_from_frames = torch.cat([frames[:1, :], frames[1:, :] - frames[:-1, :]], dim=0) == 1
        # we need to create this to only query the first element of each frame start

        for nonzero in onsets_from_frames.nonzero():
            frame = nonzero[0].item()
            pitch = nonzero[1].item()

            onset = frame
            offset = frame

            while frames[offset, pitch].item():
                offset += 1
                if offset == frames.shape[0]:  # = we reach the end of the prediction
                    break

            if offset > onset:
                pitches.append(pitch)
                intervals.append([onset, offset])
                velocities.append(70)
        return np.array(pitches), np.array(intervals), np.array(velocities)


class BpNTPrediction(ModelNTPrediction):
    # the constant names are copied from the original repository
    # therefore it might not be possible to maintain a unified naming scheme in this repository
    FFT_HOP = 256

    SAMPLE_RATE = 22050
    AUDIO_SAMPLE_RATE = 22050

    ANNOTATIONS_FPS = AUDIO_SAMPLE_RATE // FFT_HOP
    SCALING_REAL_TO_FRAME = ANNOTATIONS_FPS
    SCALING_FRAME_TO_REAL = 1.0 / ANNOTATIONS_FPS

    MIN_MIDI = 21

    DEFAULT_ONSET_THRESHOLD = 0.5
    DEFAULT_FRAME_THRESHOLD = 0.3

    def __init__(self, dataset_prediction_mapping: Dict[AmtEvalDataset, str], logger: logging.Logger):
        super().__init__(dataset_prediction_mapping, logger)

    def __str__(self):
        return "BpNTPrediction"

    def find_matching_midi_prediction(self, basename, prediction_dir) -> str:
        return super().find_matching_midi_prediction(basename, prediction_dir)

    def optimal_threshold(self) -> float:
        best_thresholds: List[float] = []

        onset_values_for_diagram = {np.round(x, decimals=2): [] for x in np.arange(0.05, 0.8, 0.05)}
        frame_values_for_diagram = {np.round(x, decimals=2): [] for x in np.arange(0.05, 0.8, 0.05)}

        for dataset, prediction_dir in self.dataset_prediction_mapping.items():
            # todo assert that .npz files exist
            for label in tqdm(dataset):
                basename: str = str(os.path.basename(label[0]).replace('.wav', ''))
                matching_npz_file: str = super().find_matching_file(basename, prediction_dir, '*.npz')
                data: np.ndarray = np.load(matching_npz_file, allow_pickle=True)['basic_pitch_model_output'].item()

                note: np.ndarray = data['note']
                contour: np.ndarray = data['contour']
                onset: np.ndarray = data['onset']

                ref = self.get_p_i_v_attributes_from_midi(label[1], note.shape, self.SCALING_REAL_TO_FRAME,
                                                          self.SCALING_FRAME_TO_REAL)

                best_threshold: float = -1
                best_threshold_value: np.float64 = np.float64(-1)

                for threshold in np.arange(0.05, 0.8, 0.05):
                    threshold = np.round(threshold, decimals=2)
                    # todo this method needs very long time to run @ small thresholds
                    # -> python profile to optimize this?
                    est = self.get_p_i_v_from_prediction(note, contour, onset, threshold, threshold)

                    if len(est['p_midi']) == 0:
                        continue

                    frame_metrics: Dict[str, float] = mir_eval.multipitch.evaluate(ref['t_time'], ref['f'],
                                                                                   est['t_time'], est['f'])
                    frame_f1 = hmean([frame_metrics['Precision'] + eps, frame_metrics['Recall'] + eps]) - eps
                    p_onset, r_onset, f_onset, o_onset = mir_eval.transcription.precision_recall_f1_overlap(
                        ref['i_time'],
                        ref['p_hz'],
                        est['i_time'],
                        est['p_hz'], offset_ratio=None)

                    onset_values_for_diagram[threshold].append(f_onset)
                    frame_values_for_diagram[threshold].append(frame_f1)

                    current_val: np.float64 = np.mean([frame_f1, f_onset])  # noqa (returns float64 not ndarray)
                    if best_threshold_value < current_val:
                        best_threshold_value = current_val
                        best_threshold = threshold
                assert best_threshold > 0
                best_thresholds.append(best_threshold)

        visualizations.plots.plot_threshold_optimization(onset_values_for_diagram, frame_values_for_diagram)

        return float(np.mean(best_thresholds))

    def calculate(self, save_path: str, **kwargs) -> Dict:
        all_metrics: Dict[str, Any] = {}

        frame_threshold = kwargs.get('frame_threshold')
        onset_threshold = kwargs.get('onset_threshold')
        calc_ap: bool = kwargs.get('calc_ap', False)

        dataset: AmtEvalDataset
        prediction_dir: str
        for dataset, prediction_dir in self.dataset_prediction_mapping.items():
            metrics: Dict[str, Any] = {}
            compute_ap_metrics: bool = True and calc_ap
            for label in tqdm(dataset):
                basename: str = str(os.path.basename(label[0]).replace('.wav', ''))
                ref_midi = label[1]

                ap_metrics = {}
                nt_metrics = {}
                nt_metrics_onsets = {}
                if compute_ap_metrics:
                    ap_metrics = self.calc_ap_values(basename, prediction_dir, label[1])

                if frame_threshold is None and onset_threshold is None:
                    matching_midi_prediction: str = self.find_matching_midi_prediction(basename, prediction_dir)
                    nt_metrics = metrics_midi_nt.calculate_metrics(matching_midi_prediction, ref_midi)
                    mpe_metrics = self.calc_metrics_and_save_midi_from_frames(basename, prediction_dir,
                                                                              self.DEFAULT_ONSET_THRESHOLD,
                                                                              self.DEFAULT_FRAME_THRESHOLD,
                                                                              ref_midi)
                elif frame_threshold is not None and onset_threshold is not None:
                    nt_metrics: Dict[str, float] = self.calc_metrics_and_save_midi(basename, prediction_dir,
                                                                                   onset_threshold, frame_threshold,
                                                                                   ref_midi)
                    mpe_metrics = self.calc_metrics_and_save_midi_from_frames(basename, prediction_dir, onset_threshold,
                                                                              frame_threshold, ref_midi)
                    nt_metrics_onsets = self.calc_onset_metrics(ref_midi, basename, prediction_dir, onset_threshold)
                else:
                    raise RuntimeError(
                        f'You need to set frame_threshold as well as onset_threshold. You need to set both.')

                joined_metrics: Dict[str, float] = {**nt_metrics, **mpe_metrics, **ap_metrics, **nt_metrics_onsets}
                for key, value in joined_metrics.items():
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

    def calc_ap_values(self, basename, prediction_dir, midipath: str) -> Dict[str, float]:
        self.logger.info(f'Calculating ap values for label: {basename}')

        matching_npz_file: str = super().find_matching_file(basename, prediction_dir, '*.npz')
        data: np.ndarray = np.load(matching_npz_file, allow_pickle=True)['basic_pitch_model_output'].item()

        note: np.ndarray = data['note']
        contour: np.ndarray = data['contour']
        onset: np.ndarray = data['onset']

        # Todo what happens if we use matching_midi_predction instead of midipath???
        frame_ap: float = self.calc_mpe_frame_ap(midipath, note)
        frame_ap_frame = self.calc_frame_ap(midipath, note)
        note_ap_frame = self.calc_note_ap(midipath, {'note': note, 'onset': onset, 'contour': contour})
        return {
            'mpe/frame-raw/avg_precision': frame_ap,
            'mpe/frame/avg_precision': frame_ap_frame,
            'nt/frame/avg_precision': note_ap_frame,
            'nt/onset-raw/avg_precision': self.calc_onset_ap(midipath, onset)
        }

    def calc_metrics_and_save_midi(self, basename, prediction_dir, onset_threshold: float, frame_threshold: float,
                                   ref_midi_path: str) -> Dict[str, float]:
        matching_npz_file: str = super().find_matching_file(basename, prediction_dir, '*.npz')
        data: np.ndarray = np.load(matching_npz_file, allow_pickle=True)['basic_pitch_model_output'].item()

        note: np.ndarray = data['note']
        contour: np.ndarray = data['contour']
        onset: np.ndarray = data['onset']

        est = self.get_p_i_v_from_prediction(note, contour, onset, onset_threshold, frame_threshold)
        threshold_optimized_midi_filepath = os.path.join(prediction_dir, f'{basename}-opt-thr.mid')
        midi.save_p_i_as_midi(threshold_optimized_midi_filepath, est['p_midi'], est['i_time'], est['v'])
        return metrics_midi_nt.calculate_metrics(threshold_optimized_midi_filepath, ref_midi_path)

    def calc_metrics_and_save_midi_from_frames(self, basename, prediction_dir, onset_threshold,
                                               frame_threshold, ref_midi_path):
        matching_npz_file: str = super().find_matching_file(basename, prediction_dir, '*.npz')
        data: np.ndarray = np.load(matching_npz_file, allow_pickle=True)['basic_pitch_model_output'].item()
        note: np.ndarray = data['note']

        est = self.get_p_i_v_from_prediction(note, onset_threshold=onset_threshold, frame_threshold=frame_threshold)
        midi_filepath = os.path.join(prediction_dir, f'{basename}-opt-thr-fromframes.mid')
        midi.save_p_i_as_midi(midi_filepath, est['p_midi'], est['i_time'], est['v'])

        return metrics_midi_nt.calculate_metrics(midi_filepath, ref_midi_path, pred_source='mpe')

    def get_p_i_v_from_prediction(self, note_prediction: np.ndarray, contour_prediction: np.ndarray = None,
                                  onset_prediction: np.ndarray = None, onset_threshold: float = 0.5,
                                  frame_threshold: float = 0.3):
        if contour_prediction is not None and onset_prediction is not None:
            midifile, note_events = basic_pitch.note_creation.model_output_to_notes(
                {'note': note_prediction, 'onset': onset_prediction, 'contour': contour_prediction},
                onset_thresh=onset_threshold, frame_thresh=frame_threshold)
            p_midi, i_time, v = midi.get_p_i_v_from_note_events(note_events)
            i_frames: np.ndarray = (i_time * self.SCALING_REAL_TO_FRAME).astype(int)
        elif note_prediction is not None and contour_prediction is None and onset_prediction is None:
            p_midi, i_frames, v = OnsetsAndFramesNTPrediction.extract_notes_from_frames(
                torch.from_numpy(note_prediction).float(), frame_threshold)
            i_time: np.ndarray = (i_frames * self.SCALING_FRAME_TO_REAL)
            p_midi = p_midi + self.MIN_MIDI
        else:
            raise RuntimeError(f'Unknown combination of inputs.')

        p_hz: np.ndarray = np.array([mir_eval.util.midi_to_hz(p) for p in p_midi])

        t, f = self.notes_to_frames(p_midi, i_frames, note_prediction.shape)
        t_time = t.astype(np.float64) * self.SCALING_FRAME_TO_REAL

        return {
            'p_midi': p_midi,
            'p_hz': p_hz,
            'i_time': i_time,
            'i_frames': i_frames,
            'v': v,
            't': t,
            't_time': t_time,
            'f': f
        }

    def calc_note_ap(self, midi_path: str, bp_model_output: Dict):
        """
        Calculates AP values based on bp model output (final output)
        Args:
            midi_path:
            bp_model_output:

        Returns:

        """
        # todo this might be wrong! (using bp_model_output...shape)
        ref = self.get_p_i_v_attributes_from_midi(midi_path, bp_model_output['note'].shape, self.SCALING_REAL_TO_FRAME,
                                                  self.SCALING_FRAME_TO_REAL)
        precision_recall_pairs_frame: List[Tuple[float, float]] = []
        thresholds = []
        # starting at 0.1 because basic_pitch.note_creation.model_output_to_notes is very bad with low threshold
        # todo investigate this
        for threshold in np.arange(0.1, 1.0, 0.05):
            est = self.get_p_i_v_from_prediction(bp_model_output['note'], bp_model_output['contour'],
                                                 bp_model_output['onset'], onset_threshold=threshold,
                                                 frame_threshold=threshold)

            if len(est['p_midi']) == 0:
                continue

            frame_metrics: Dict[str, float] = mir_eval.multipitch.evaluate(ref['t_time'], ref['f'], est['t_time'],
                                                                           est['f'])
            p_frame: float = frame_metrics['Precision']
            r_frame: float = frame_metrics['Recall']
            thresholds.append(threshold)
            precision_recall_pairs_frame.append((p_frame, r_frame))
        return ap.calc_ap_from_prec_recall_pairs(precision_recall_pairs_frame, plot=False, thresholds=thresholds,
                                                 title='Frame Precision/Recall Curve')

    def calc_frame_ap(self, midi_path: str, note: np.ndarray) -> float:
        """
        In this evaluation, we convert the frame output back to note events.
        Calculates ap values based out of model frame output.
        Args:
            midi_path: Reference midi path
            note: frame probabilities output from basic-pitch
        Returns: ap for frame, onset, onset-offset
        """
        # todo this might be wrong! (using note.shape)
        ref = self.get_p_i_v_attributes_from_midi(midi_path, note.shape, self.SCALING_REAL_TO_FRAME,
                                                  self.SCALING_FRAME_TO_REAL)

        precision_recall_pairs_frame: List[Tuple[float, float]] = []

        executed_thresholds = []  # Just used for visualization purposes
        for threshold in np.arange(0, 1.0, 0.05):
            est = self.get_p_i_v_from_prediction(note, frame_threshold=threshold)

            if len(est['p_midi']) == 0:
                # This happens when the threshold is either very small or very large
                # Then we cannot reliably compute all the metrics -> weird values -> skipping this
                continue

            frame_metrics: Dict[str, float] = mir_eval.multipitch.evaluate(ref['t_time'], ref['f'], est['t_time'],
                                                                           est['f'])
            p_frame: float = frame_metrics['Precision']
            r_frame: float = frame_metrics['Recall']

            executed_thresholds.append(threshold)
            precision_recall_pairs_frame.append((p_frame, r_frame))

        return ap.calc_ap_from_prec_recall_pairs(precision_recall_pairs_frame, plot=False,
                                                 thresholds=executed_thresholds,
                                                 title='Frame Precision/Recall Curve')

    def calc_mpe_frame_ap(self, midi_path: str, note: np.ndarray) -> float:
        """
        Calculates the frame average precision based on the raw frame output (note) of the bp model.
        Calculates the AP using sklearn
        Args:
            midi_path: path to reference midi file
            note: shape(num_of_frames, 88)
        Returns: AP value for this metric
        """
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
        avg_precision_score = sk_metrics.average_precision_score(f_annot_pitch.flatten(), note.flatten())
        return avg_precision_score

    def calc_onset_ap(self, midi_path: str, onset_prediction: np.ndarray, plot=False):
        note_events = utils.midi.parse_midi_note_tracking(midi_path)

        columns_before = np.zeros((onset_prediction.shape[0], 21), dtype=onset_prediction.dtype)
        columns_after = np.zeros((onset_prediction.shape[0], 19), dtype=onset_prediction.dtype)
        onset_prediction = np.concatenate((columns_before, onset_prediction), axis=1)
        onset_prediction = np.concatenate((onset_prediction, columns_after), axis=1)

        f_annot_onsets = (
            metrics_prediction.metrics_prediction_nt.compute_onset_array_nooverlap(note_events,
                                                                                   onset_prediction.shape[0],
                                                                                   self.SCALING_REAL_TO_FRAME,
                                                                                   'pitch').T)
        avg_precision_score = sk_metrics.average_precision_score(f_annot_onsets.flatten(), onset_prediction.flatten())
        if plot:
            PrecisionRecallDisplay.from_predictions(f_annot_onsets.flatten(), onset_prediction.flatten())
            plt.show()
        return avg_precision_score

    def calc_onset_metrics(self, midi_path: str, basename, prediction_dir, onset_threshold: float) -> Dict[
        str, float]:

        matching_npz_file: str = super().find_matching_file(basename, prediction_dir, '*.npz')
        data: np.ndarray = np.load(matching_npz_file, allow_pickle=True)['basic_pitch_model_output'].item()

        onset_prediction: np.ndarray = data['onset']

        onset_prediction = (onset_prediction > onset_threshold)

        note_events = utils.midi.parse_midi_note_tracking(midi_path)

        columns_before = np.zeros((onset_prediction.shape[0], 21), dtype=onset_prediction.dtype)
        columns_after = np.zeros((onset_prediction.shape[0], 19), dtype=onset_prediction.dtype)
        onset_prediction = np.concatenate((columns_before, onset_prediction), axis=1)
        onset_prediction = np.concatenate((onset_prediction, columns_after), axis=1)

        f_annot_onsets = (
            metrics_prediction.metrics_prediction_nt.compute_onset_array_nooverlap(note_events,
                                                                                   onset_prediction.shape[0],
                                                                                   self.SCALING_REAL_TO_FRAME,
                                                                                   'pitch').T)
        precision = sk_metrics.precision_score(f_annot_onsets.flatten(), onset_prediction.flatten())
        recall = sk_metrics.recall_score(f_annot_onsets.flatten(), onset_prediction.flatten())
        f1 = sk_metrics.f1_score(f_annot_onsets.flatten(), onset_prediction.flatten())
        return {
            'nt/onset-raw/precision': precision,
            'nt/onset-raw/recall': recall,
            'nt/onset-raw/f1': f1
        }
