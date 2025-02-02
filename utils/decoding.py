from typing import Tuple, List

import mir_eval.util
import numpy as np

import constants


# def extract_notes(onsets, frames, velocity, onset_threshold=0.5, frame_threshold=0.5):
#     """
#     Finds the note timings based on the onsets and frames information
#
#     Parameters
#     ----------
#     onsets: torch.FloatTensor, shape = [frames, bins]
#     frames: torch.FloatTensor, shape = [frames, bins]
#     velocity: torch.FloatTensor, shape = [frames, bins]
#     onset_threshold: float
#     frame_threshold: float
#
#     Returns
#     -------
#     pitches:    np.ndarray of bin_indices
#                 shape: (<length>, 1)
#                 To my understanding, these are the pitch values for each time index
#     intervals:  np.ndarray of rows containing (onset_index, offset_index)
#                 shape: (<length>, 2)
#                 Start and end of each note
#     velocities: np.ndarray of velocity vales
#                 shape: (<length>, 1)
#                 Velocity value for each time index
#     """
#     onsets = (onsets > onset_threshold).cpu().to(torch.uint8)
#     frames = (frames > frame_threshold).cpu().to(torch.uint8)
#     # torch.cat = concatenates tensors. Requirement: each tensor has the same shape!
#     # onsets[:1, :] = first row, keeping all columns (=time bin 0 with all possible key values)
#     # onsets[1:, :] - onsets[:-1, :] = subtracts each row of onsets from the next row, creating the difference
#     # This is true if the current index detects an onset and the next index does not.
#     onset_diff = torch.cat([onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], dim=0) == 1
#
#     pitches = []
#     intervals = []
#     velocities = []
#
#     for nonzero in onset_diff.nonzero():
#         frame = nonzero[0].item()
#         pitch = nonzero[1].item()
#
#         onset = frame
#         offset = frame
#         velocity_samples = []
#
#         while onsets[offset, pitch].item() or frames[offset, pitch].item():
#             if onsets[offset, pitch].item():
#                 velocity_samples.append(velocity[offset, pitch].item())
#             offset += 1
#             if offset == onsets.shape[0]:
#                 break
#
#         if offset > onset:
#             pitches.append(pitch)
#             intervals.append([onset, offset])
#             velocities.append(np.mean(velocity_samples) if len(velocity_samples) > 0 else 0)
#
#     return np.array(pitches), np.array(intervals), np.array(velocities)


def notes_to_frames(pitches_midi, intervals) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Takes lists specifying notes and intervals. Returns active pitches in each frame.
    -> This is required for multipitch evaluation!!!

    Parameters
    ----------
    pitches_midi: list of midi pitches
    intervals: list of [onset, offset] time for each pitch

    Returns
    -------
    time: np.ndarray containing the frame indices, shape(n_frames,1)
    freqs: list of np.ndarray, each containing the frequency bin indices, shape(n_frames,[list of frequencies])
    """
    roll = np.zeros((max(max(row) for row in intervals), constants.MAX_MIDI))
    for pitch, (onset, offset) in zip(pitches_midi, intervals):
        roll[onset:offset, pitch] = 1

    time = np.arange(roll.shape[0])
    freqs = [roll[t, :].nonzero()[0] for t in time]
    return time, freqs


def note_to_multipitch_realtime(pitches: np.ndarray, intervals: np.ndarray,
                                scaling_frame_to_real: float) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Required parameters for mir_eval.multipitch:
    ref_time = reference time stamps in seconds, shape(n,1) = list of timestamps realtime!
    ref_freqs = reference frequencies in Hz, list of np.ndarray = list of active frequencies for each timestamp

    This method is required because the original metric computation does use realtime for the evaluation.
    This is an issue with mir_eval because they expect realtime, not frame time intervals.
    Theoretically it's no issue but with larger files, they have hardcoded a limit where it's not possible anymore to
    calculate metrics.

    :param pitches: ndarray of pitch values in midi format, shape(n,1)
    :param intervals: onset and offset time specified for each pitch value, shape(n,2)
    :param scaling_frame_to_real: scaling value.

    :return: time_seconds, freqs
    """

    time, active_midis = notes_to_frames(pitches, intervals)
    time_seconds = time * scaling_frame_to_real
    freqs = [mir_eval.util.midi_to_hz(midis) for midis in active_midis]
    return time_seconds, freqs
