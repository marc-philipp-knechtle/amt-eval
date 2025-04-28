"""
Note Tracking Metrics for midi files
"""
import itertools
import sys
from typing import List, Tuple, Dict

import mir_eval
import numpy as np
from scipy.stats import hmean

from constants import HOP_LENGTH, SAMPLE_RATE
from utils import midi, decoding

eps = sys.float_info.epsilon

scaling_frame_to_real = HOP_LENGTH / SAMPLE_RATE
"""
With scaling_frame_to_real, we can convert from time bin indices back to realtime
"""
scaling_real_to_frame = SAMPLE_RATE / HOP_LENGTH
"""
With scaling_real_to_frame, we can convert from realtime to bin indices
"""


def calculate_metrics(prediction_filepath: str, source_filepath: str, pred_source: str = 'nt') -> Dict[str, float]:
    """
    Calculates Note Tracking metrics based on two midi files.
    Args:
        prediction_filepath: prediction midi
        source_filepath: source midi
        pred_source: 'nt' or 'mpe'
    Returns: metrics in Dictionary format
    """
    prediction_note_tracking: np.ndarray = midi.parse_midi_note_tracking(prediction_filepath)

    pitches_est: List[int] = []
    intervals_est: List[Tuple[float, float]] = []
    velocities_est: List[int] = []
    for start_time, end_time, pitch, velocity in prediction_note_tracking:
        if start_time == end_time:
            continue
        pitches_est.append(int(pitch))
        intervals_est.append((float(start_time), float(end_time)))
        velocities_est.append(velocity)

    p_est: np.ndarray = np.array(pitches_est)
    """
    Array of estimated pitches (in midi values)
    shape=(n,)
    """
    p_est_hz: np.ndarray = np.array([mir_eval.util.midi_to_hz(p) for p in p_est])
    """
    shape=(n,)
    """
    i_est: np.ndarray = np.array(intervals_est).reshape(-1, 2)
    """
    List of estimated intervals (onset time, offset time), in real! time
    shape=(n,2)
    """
    v_est: np.ndarray = np.array(velocities_est)
    """
    shape=(n,)
    """

    label_note_tracking: np.ndarray = midi.parse_midi_note_tracking(source_filepath)
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

    metrics = {}
    p = round(p, 3)
    r = round(r, 3)
    f = round(f, 3)
    o = round(o, 3)
    metrics[f'{pred_source}/note/precision'] = p
    metrics[f'{pred_source}/note/recall'] = r
    metrics[f'{pred_source}/note/f1'] = f
    # metrics[f'{pred_source}/note/overlap'] = o

    p, r, f, o = mir_eval.transcription.precision_recall_f1_overlap(i_ref, p_ref_hz, i_est, p_est_hz)
    metrics[f'{pred_source}/note-with-offsets/precision'] = p
    metrics[f'{pred_source}/note-with-offsets/recall'] = r
    metrics[f'{pred_source}/note-with-offsets/f1'] = f
    # metrics[f'{pred_source}/note-with-offsets/overlap'] = o

    # p, r, f, o = mir_eval.transcription_velocity.precision_recall_f1_overlap(i_ref, p_ref_hz, v_ref,
    #                                                                          i_est, p_est_hz, v_est,
    #                                                                          offset_ratio=None,
    #                                                                          velocity_tolerance=0.1)
    # metrics[f'{pred_source}/note-with-velocity/precision'] = p
    # metrics[f'{pred_source}/note-with-velocity/recall'] = r
    # metrics[f'{pred_source}/note-with-velocity/f1'] = f
    # metrics[f'{pred_source}/note-with-velocity/overlap'] = o

    # p, r, f, o = mir_eval.transcription_velocity.precision_recall_f1_overlap(i_ref, p_ref_hz, v_ref,
    #                                                                          i_est, p_est_hz, v_est,
    #                                                                          velocity_tolerance=0.1)
    # metrics[f'{pred_source}/note-with-offsets-and-velocity/precision'] = p
    # metrics[f'{pred_source}/note-with-offsets-and-velocity/recall'] = r
    # metrics[f'{pred_source}/note-with-offsets-and-velocity/f1'] = f
    # metrics[f'{pred_source}/note-with-offsets-and-velocity/overlap'] = o

    frame_metrics = evaluate_note_based_mpe(p_ref, i_ref, p_est, i_est)
    for key, loss in frame_metrics.items():
        metrics[f'{pred_source}/frame/' + key.lower().replace(' ', '_')] = loss
    metrics[f'{pred_source}/frame/f1'] = (
            hmean([frame_metrics['Precision'] + eps, frame_metrics['Recall'] + eps]) - eps)

    del i_ref, i_est, p_est, p_ref, p_est_hz, p_ref_hz
    return metrics


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

    frame_metrics = mir_eval.multipitch.evaluate(t_ref, f_ref, t_est, f_est)
    frame_metrics_filtered = {key: frame_metrics[key] for key in ['Precision', 'Recall'] if key in frame_metrics}
    return frame_metrics_filtered
