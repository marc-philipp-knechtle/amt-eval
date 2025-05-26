import numpy as np
import torch


def compute_annotation_array_nooverlap_from_midi(midifile: str):
    ...


def compute_onset_array_nooverlap(note_events, num_time_frames, fs, annot_type='pitch_class', shorten=1.0):
    """
    This tries to be the equivalent function of the same func. for frame

    Args:
        note_events:
        num_time_frames:
        fs:
        annot_type:
        shorten:

    Returns:

    """
    array_length = num_time_frames

    if annot_type == 'pitch_class':
        array_height = 12
    elif annot_type == 'pitch':
        array_height = 128
    elif annot_type == 'piano':
        array_height = 88
    elif annot_type == 'instruments':
        array_height = 1
    else:
        assert False, ['annotation type ' + str(annot_type) + ' not valid!']

    annot_array = np.zeros((array_height, array_length))

    note_events_cp = note_events.copy()
    note_events_cp[:, :2] = np.floor(note_events_cp[:, :2] * fs).astype(int)

    for line_num in range(note_events_cp.shape[0]):
        start_ind = int(note_events_cp[line_num, 0])

        if annot_type == 'pitch_class':
            pitch_ind = int(np.mod(note_events_cp[line_num, 2], 12))
        elif annot_type == 'pitch':
            pitch_ind = int(note_events_cp[line_num, 2])
        elif annot_type == 'instruments':
            pitch_ind = 0
        elif annot_type == 'piano':
            pitch_ind = int(note_events_cp[line_num, 2])
        else:
            raise RuntimeError(f'Unexpected annotation type: {annot_type}')

        annot_array[pitch_ind, start_ind - 2:start_ind + 1] = 1

    return annot_array


def compute_annotation_array_nooverlap(note_events, num_time_frames, fs, annot_type='pitch_class', shorten=1.0):
    """ Converts a note event list into a binary np array, assuming a given frame rate
    From cweiss -> multipitch_architectures

    the output array length has the number of time_frames form the computed hcqt -> matches the length of the audio
    (if we give the length of the audio to num_time_frames)
    If note_events is longer than num_time_frames the note event is cut off.

    Args:
        note_events:        np array of note events 'start_sec', 'end_sec', 'pitchclass', 'MIDI_channel'
        num_time_frames:    number of time bins
        fs:                 simply the sampling rate
        annot_type:         type of third column: 'pitch' (MIDI pitch) or 'pitch_class' (0...11)
        shorten:            Fraction of duration for shortening note events

    Returns:
        annot_array:       np array containing binary pitch activity, dimensions "#pitch_bins * #time_frames
    """

    if annot_type == 'pitch_class':
        array_height = 12
    elif annot_type == 'pitch':
        array_height = 128
    elif annot_type == 'piano':
        array_height = 88
    elif annot_type == 'instruments':
        array_height = 1
    else:
        assert False, ['annotation type ' + str(annot_type) + ' not valid!']

    array_length = num_time_frames
    """
    length of the audio
    """

    annot_array = np.zeros((array_height, array_length))

    # if shorten != 1.0:
    #     note_events[:, 1] = note_events[:, 0] + shorten * (note_events[:, 1] - note_events[:, 0])

    note_events_frameinds = note_events.copy()
    note_events_frameinds[:, :2] = np.floor(note_events_frameinds[:, :2] * fs).astype(int)
    """
    We convert here the start and end times of the notes to frame indices
    (multiply the start and end times with scaling) 
    """
    durations = note_events_frameinds[:, 1] - note_events_frameinds[:, 0]
    """
    shape = num_of_notes
    Note durations in frametime
    """
    vanishing_events = np.array(np.nonzero((durations < 1).astype(int))).squeeze()
    """
    seems very unlikely, because we use frametime, durations smaller than 1 should not exist
    Because this does very likely not contain anything, we can ignore the output from here. 
    """
    vanishing_endtimes = np.unique(note_events_frameinds[vanishing_events, 1])
    for vind in range(vanishing_endtimes.shape[0]):
        note_events_frameinds[np.where(note_events_frameinds[:, 0] == vanishing_endtimes[vind])[0], 0] += 1
        note_events_frameinds[np.where(note_events_frameinds[:, 1] == vanishing_endtimes[vind])[0], 1] += 1

    note_events_frameinds[vanishing_events, 0] -= 1
    durations_new = note_events_frameinds[:, 1] - note_events_frameinds[:, 0]
    vanishing_events_new = np.array(np.nonzero((durations_new < 1).astype(int))).squeeze()
    note_events_frameinds[vanishing_events_new, 0] -= 1

    # check if events of length<1 still exist
    durations_new = note_events_frameinds[:, 1] - note_events_frameinds[:, 0]
    assert np.nonzero((durations_new < 1).astype(int))[0].size == 0, 'still events of length<1 after correction!'

    for line_num in range(note_events_frameinds.shape[0]):
        start_ind = int(note_events_frameinds[line_num, 0])
        end_ind = int(note_events_frameinds[line_num, 1])

        if annot_type == 'pitch_class':
            pitch_ind = int(np.mod(note_events_frameinds[line_num, 2], 12))
        elif annot_type == 'pitch':
            pitch_ind = int(note_events_frameinds[line_num, 2])
        elif annot_type == 'instruments':
            pitch_ind = 0
        elif annot_type == 'piano':
            pitch_ind = int(note_events_frameinds[line_num, 2])
        else:
            raise RuntimeError(f'Unexpected annotation type: {annot_type}')

        annot_array[pitch_ind, start_ind:end_ind] = 1

    return annot_array
