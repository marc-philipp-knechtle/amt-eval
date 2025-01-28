import logging
import os.path
import re
import sys
from abc import abstractmethod
from glob import glob
from typing import List, Dict, Tuple

import librosa
import numpy as np
import pandas as pd
import pretty_midi
from torch.utils.data import Dataset
from torch import Tensor
import soundfile
import torch

from constants import SAMPLE_RATE, MAX_MIDI, MIN_MIDI, HOP_LENGTH, HOPS_IN_ONSET, HOPS_IN_OFFSET
from utils import midi, log

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = log.CustomFormatter()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class NoteTrackingDataset(Dataset):
    path: str
    groups: List[str]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data: List[Dict[str, Tensor]]
    """
    Tuple[filename,Tensor=basically the .pt files which are saved automatically] 
    """

    def __init__(self, path: str, groups: List[str] = None, logging_filepath: str = None):
        self.path = path
        self.groups = groups if groups is not None else self.available_groups()
        self.data: List[Dict[str, Tensor]] = []
        """
        data for this dataset
        List of Dict[audio_filepath, Tensordata(see function load)]
        """

        if logging_filepath is not None:
            file_handler = logging.FileHandler(logging_filepath)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.INFO)
            logger.addHandler(file_handler)

        logger.info(f"Loading {len(self.groups)} group{'s' if len(self.groups) > 1 else ''} "
                    f"of {self.__class__.__name__} at {self.path}")

        for group in self.groups:
            input_file: Tuple[str, str]
            """
            [str, str] = ['path to audio', 'path to annotation']
            """
            for input_file in self.get_files(group):
                self.data.append(self.load(input_file[0], input_file[1]))

    def __getitem__(self, index):
        """
        label = used to store the information efficiently (see load())
        :return: Dictionary of saved item: path, audio, label, velocity, onset, offset, frame
        """
        data: Dict[str, Tensor] = self.data[index]
        result = (
            dict(path=data['path'],
                 audio=data['audio'].to(self.device).float().div_(32768.0),
                 label=data['label'].to(self.device),
                 velocity=data['velocity'].to(self.device).float().div_(128.0),
                 onset=(data['label'].to(self.device) == 3).float(),
                 offset=(data['label'].to(self.device) == 1).float(),
                 frame=(data['label'].to(self.device) > 1).float()))
        return result

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for i in range(len(self.data)):
            yield self[i]

    @classmethod
    @abstractmethod
    def available_groups(cls) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def get_files(self, group: str) -> List[Tuple[str, str]]:
        """
        :param group: group
        :return: List of Tuple[audio_filename, tsv_filename]
        tsv is a list representation in the form of: [onset,offset,pitch,velocity]
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def load_annotations(annotation_path: str) -> np.ndarray:
        """
        used in NoteTrackingDataset.load(...)
        :return: annotation as np.ndarray in the form of [onset,offset,pitch,velocity]
        """
        raise NotImplementedError

    def load(self, audio_path: str, annotation_path: str) -> Dict[str, Tensor]:
        """
        load an audio track and the corresponding annotations

        Returns
        -------
            A dictionary containing the following items:

            path: str
                the path to the audio file

            audio: torch.ShortTensor, shape = [num_frames]
                the raw waveform

            label: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains the onset/offset/frame labels encoded as:
                3 = onset, 2 = frames after onset, 1 = offset, 0 = all else

            velocity: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains MIDI velocity values at the frame locations
        """
        saved_data_path = audio_path.replace('.flac', '.pt').replace('.wav', '.pt')
        if os.path.exists(saved_data_path):
            return torch.load(saved_data_path)

        audio: np.ndarray
        audio, sr = soundfile.read(audio_path, dtype='int16', always_2d=True)
        # Conversion to float see:
        # https://stackoverflow.com/questions/58810035/converting-audio-files-between-pydub-and-librosa
        audio: np.ndarray = np.array(audio).astype(np.float32)
        # converting from 2D representations (left and right channel) to 1D
        audio = audio.T
        audio = audio[0]

        if sr != SAMPLE_RATE:
            logger.info(f"Sample Rate Mismatch: resampling file: {audio_path}, from {sr} to default: {SAMPLE_RATE}")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

        audio_tensor = torch.ShortTensor(audio)
        audio_length = len(audio_tensor)

        n_keys = MAX_MIDI - MIN_MIDI + 1
        n_steps = (audio_length - 1) // HOP_LENGTH + 1

        label = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
        velocity = torch.zeros(n_steps, n_keys, dtype=torch.uint8)

        annotations: np.ndarray = self.load_annotations(annotation_path)

        for onset, offset, note, vel in annotations:
            left = int(round(onset * SAMPLE_RATE / HOP_LENGTH))
            onset_right = min(n_steps, left + HOPS_IN_ONSET)
            frame_right = int(round(offset * SAMPLE_RATE / HOP_LENGTH))
            frame_right = min(n_steps, frame_right)
            offset_right = min(n_steps, frame_right + HOPS_IN_OFFSET)

            f = int(note) - MIN_MIDI
            label[left:onset_right, f] = 3
            label[onset_right:frame_right, f] = 2
            label[frame_right:offset_right, f] = 1
            velocity[left:frame_right, f] = vel

        data: Dict[str, Tensor] = dict(path=audio_path, audio=audio_tensor, label=label, velocity=velocity)
        torch.save(data, saved_data_path)
        return data


class SchubertWinterreiseDataset(NoteTrackingDataset):
    swd_midi: str
    swd_csv: str
    swd_tsv: str
    swd_audio_wav: str

    def __init__(self, path='datasets/Schubert_Winterreise_Dataset_v2-1', groups=None, logger_filepath: str = None):
        # adding underscore to symbolize that these annotations are computationally created
        self.swd_midi = os.path.join(path, '02_Annotations', '_ann_audio_note_midi')
        self.swd_csv = os.path.join(path, '02_Annotations', 'ann_audio_note')
        self.swd_tsv = os.path.join(path, '02_Annotations', '_ann_audio_note_tsv')
        self.swd_audio_wav = os.path.join(path, '01_RawData', 'audio_wav')

        super().__init__(path, groups, logger_filepath)

    def __str__(self):
        return 'SchubertWinterreiseDataset'

    @staticmethod
    def get_filepaths_for_group(directory: str, group_pattern) -> List[str]:
        """
        Returns: All matching files (paths) for a certain group
        """
        files = glob(os.path.join(directory, '**', '*.wav'), recursive=True)
        matching_files = [file for file in files if re.compile(fr".*{re.escape(group_pattern)}.*").search(file)]
        return matching_files

    @classmethod
    def available_groups(cls) -> List[str]:
        """
        HU33, SC06 are the public datasets -> these are used preferred for testing
        Returns: Available groups
        """
        return ['AL98', 'FI55', 'FI66', 'FI80', 'HU33', 'OL06', 'QU98', 'SC06', 'TR99']

    def get_files(self, group: str) -> List[Tuple[str, str]]:
        """
        Base methods to load all audio files into memory
        Returns: List of Tuple[audio_filename.wav,midi_filename.wav]
        """
        audio_filepaths: List[str] = self.get_filepaths_for_group(self.swd_audio_wav, group)
        if len(audio_filepaths) == 0:
            raise RuntimeError(f'Expected files for group {group}, found nothing.')

        ann_audio_note_filepaths_csv: List[str] = glob(os.path.join(self.swd_csv, '*.csv'))
        assert len(ann_audio_note_filepaths_csv) > 0

        # save csv as midi
        midi_path = midi.save_csv_as_midi(ann_audio_note_filepaths_csv, self.swd_midi)
        midi_filepaths: List[str] = glob(os.path.join(midi_path, '*.mid'))

        # combine .wav with .midi
        filepaths_audio_midi: List[Tuple[str, str]] = self._combine_audio_midi(audio_filepaths, midi_filepaths)

        return self.create_audio_tsv(filepaths_audio_midi, self.swd_tsv)

    @staticmethod
    def create_audio_tsv(filepaths_audio_midi: List[Tuple[str, str]], tsv_dir: str) -> List[Tuple[str, str]]:
        """
        Creates .tsv files based on midi files (using midi.create_tsv_from_midi(...))
        Returns: List[Tuple[str, str]] of audio filepath with tsv filepath
        """
        filepaths_audio_tsv: List[Tuple[str, str]] = []
        audio_filepath: str
        midi_filepath: str
        if not os.path.exists(tsv_dir):
            os.makedirs(tsv_dir)
        for audio_filepath, midi_filepath in filepaths_audio_midi:
            tsv_filepath: str = os.path.join(tsv_dir, os.path.basename(midi_filepath).replace('.mid', '.tsv'))
            if not os.path.exists(tsv_filepath):
                midi.save_midi_as_tsv(midi_filepath, tsv_filepath)
            filepaths_audio_tsv.append((audio_filepath, tsv_filepath))
        return filepaths_audio_tsv

    @staticmethod
    def load_annotations(annotation_path: str) -> np.ndarray:
        """
        :param annotation_path: tsv annotation path
        :return: tsv as np.ndarray
        """
        return np.loadtxt(annotation_path, delimiter='\t', skiprows=1)

    @staticmethod
    def _combine_audio_midi(audio_filepaths: List[str], midi_filepaths: List[str]) -> List[Tuple[str, str]]:
        """
        Args:
            audio_filepaths: List of all audio filenames
            midi_filepaths: List of all midi filenames
        Returns: audio - midi filename combination in the form of a List of tuples
        """
        audio_midi_combination: List[Tuple[str, str]] = []
        for audio_filepath in audio_filepaths:
            basename = os.path.basename(audio_filepath)
            number_str: str = basename[14:16]
            performance: str = basename[17:21]
            # Find matching midi file
            matching_files = [midi_file for midi_file in midi_filepaths if
                              re.compile(fr".*-{number_str}_{performance}.*").search(midi_file)]
            if len(matching_files) > 1:
                raise RuntimeError(f"Found more than one matching file for audio filename: {audio_filepath}")
            midi_filepath: str = matching_files[0]
            # Create tuple
            audio_midi_combination.append((audio_filepath, midi_filepath))
        return audio_midi_combination


class WagnerRingDataset(NoteTrackingDataset):
    """
    The Wagner Ring dataset matches the Schubert Winterreise dataset in terms of audio structure. 
    However it is not easily possible to use the implementations of SWD (without SWD rewriting) as a lot of it is also SWD specific. 
    One goal would be to reuse static methods as much as possible.
    """
    wr_midi: str
    wr_csv: str
    wr_tsv: str
    wr_audio_wav: str

    def __init__(self, path='datasets/WagnerRing_v0-1', groups=None, logger_filepath: str = None):
        # adding underscore to symbolize that these annotations are computationally created
        self.wr_midi = os.path.join(path, '02_Annotations', '_ann_audio_note_midi')
        self.wr_csv = os.path.join(path, '02_Annotations', 'ann_audio_note')
        self.wr_tsv = os.path.join(path, '02_Annotations', '_ann_audio_note_tsv')
        self.wr_audio_wav = os.path.join(path, '01_RawData', 'audio_wav')

        super().__init__(path, groups, logger_filepath)

    def __str__(self):
        return 'WagnerRingDataset'

    @classmethod
    def available_groups(cls) -> List[str]:
        return ['KeilberthFurtw1952', 'Furtwangler1953', 'Krauss1953', 'Solti1958', 'Karajan1966', 'Bohm1967',
                'Swarowsky1968', 'Boulez1980', 'Janowski1980', 'Levine1987', 'Haitink1988', 'Sawallisch1989',
                'Barenboim1991', 'Neuhold1993', 'Weigle2010', 'Thielemann2011']

    def get_files(self, group: str) -> List[Tuple[str, str]]:
        """
        :return: List of audio filepath with tsv filepath
        """
        logger.info(f"Loading Files for group {group}, searching in {self.wr_audio_wav}")
        audio_filepaths: List[str] = SchubertWinterreiseDataset.get_filepaths_for_group(self.wr_audio_wav, group)
        if len(audio_filepaths) == 0:
            raise RuntimeError(f'Expected files for group {group}, found nothing.')

        ann_audio_note_filepaths_csv: List[str] = glob(os.path.join(self.wr_csv, '*.csv'))
        assert len(ann_audio_note_filepaths_csv) > 0

        # save csv as midi
        midi_path = self._save_csv_as_midi(ann_audio_note_filepaths_csv, self.wr_midi)
        midi_filepaths: List[str] = glob(os.path.join(midi_path, '*.mid'))

        # combine .wav with .midi
        filepaths_audio_midi: List[Tuple[str, str]] = self._combine_audio_midi(audio_filepaths, midi_filepaths)

        return SchubertWinterreiseDataset.create_audio_tsv(filepaths_audio_midi, self.wr_tsv)

    @staticmethod
    def _combine_audio_midi(audio_filepaths: List[str], midi_filepaths: List[str]) -> List[Tuple[str, str]]:
        audio_midi_combination: List[Tuple[str, str]] = []
        for audio_filepath in audio_filepaths:
            basename = os.path.basename(audio_filepath).replace('.wav', '')
            matching_files = [midi_file for midi_file in midi_filepaths if
                              re.compile(fr".*{basename}.*").search(midi_file)]
            if len(matching_files) != 1:
                raise RuntimeError(f"Found different number of matching midi files than expected for: {audio_filepath}")
            midi_filepath: str = matching_files[0]
            audio_midi_combination.append((audio_filepath, midi_filepath))
        return audio_midi_combination

    @staticmethod
    def load_annotations(annotation_path: str) -> np.ndarray:
        return SchubertWinterreiseDataset.load_annotations(annotation_path)

    @staticmethod
    def _save_csv_as_midi(csv_filenames: List[str], midi_path: str):
        if not os.path.exists(midi_path):
            os.mkdir(midi_path)
        for csv_filename in csv_filenames:
            ann_audio_note: pd.DataFrame = pd.read_csv(csv_filename, sep=';')
            ann_audio_filepath = os.path.join(midi_path, os.path.basename(csv_filename.replace('.csv', '.mid')))

            if os.path.exists(ann_audio_filepath):
                continue

            piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
            piano = pretty_midi.Instrument(program=piano_program)

            for _, row in ann_audio_note.iterrows():
                onset: float = row['start']
                offset: float = row['end']
                pitch: int = int(row['pitch'])

                note = pretty_midi.Note(start=onset, end=offset, pitch=pitch, velocity=64)
                piano.notes.append(note)

            file: pretty_midi.PrettyMIDI = pretty_midi.PrettyMIDI()
            file.instruments.append(piano)
            file.write(ann_audio_filepath)
        return midi_path


class Bach10Dataset(NoteTrackingDataset):
    """
    The Bach10 dataset matches SWD and WR in terms of audio structure.
    The Bach10 dataset also contains instrument specific annotations and can be used therefore as a Note Streaming
    dataset too.
    """

    bach10_midi: str
    bach10_csv: str
    bach10_tsv: str
    bach10_audio_wav: str

    def __init__(self, path='datasets/Bach10', groups=None, logger_filepath: str = None):
        # underscore = directories are computationally created and can be deleted without worrying
        self.bach10_midi = os.path.join(path, '_ann_audio_note_midi')
        self.bach10_csv = os.path.join(path, 'ann_audio_pitch_CSV')
        self.bach10_tsv = os.path.join(path, '_ann_audio_note_tsv')
        self.bach10_audio_wav = os.path.join(path, 'audio_wav_44100_mono')

        super().__init__(path, groups, logger_filepath)

    @classmethod
    def available_groups(cls) -> List[str]:
        """
        :return: just one group because Bach10 has no different groups.
        """
        return ["Bach10"]

    def get_files(self, group: str) -> List[Tuple[str, str]]:
        """
        :return: List of Tuple[audio_filename.wav, midi_filename.wav]
        """
        if group != "Bach10":
            raise RuntimeError(
                f'Group {group} not found. Bach10 supports only one single group. (specified by Bach10 group).')
        logger.info(f"Loading Files for group {group}, searching in {self.bach10_audio_wav}")

        audio_filepaths: List[str] = glob(os.path.join(self.bach10_audio_wav, '*.wav'), recursive=False)
        if len(audio_filepaths) == 0:
            raise RuntimeError(f'Expected files for group {group}, found nothing.')

        ann_audio_note_filepaths_csv: List[str] = glob(os.path.join(self.bach10_csv, '*.csv'))
        assert len(ann_audio_note_filepaths_csv) > 0

        # save csv as midi
        midi_path = midi.save_nt_csv_as_midi(ann_audio_note_filepaths_csv, self.bach10_midi)
        midi_filepaths: List[str] = glob(os.path.join(midi_path, '*.mid'))

        # combine .wav with .mid
        filepaths_audio_midi: List[Tuple[str, str]] = WagnerRingDataset._combine_audio_midi(audio_filepaths,
                                                                                            midi_filepaths)

        return SchubertWinterreiseDataset.create_audio_tsv(filepaths_audio_midi, self.bach10_tsv)

    @staticmethod
    def load_annotations(annotation_path: str) -> np.ndarray:
        return SchubertWinterreiseDataset.load_annotations(annotation_path)


class PhenicxAnechoicDataset(NoteTrackingDataset):
    """
    This implementation currently does not consider the option of having single instruments as groups.
    """
    phenicx_anechoic_midi: str
    phenicx_anechoic_tsv: str
    phenicx_anechoic_mixaudio_wav: str
    phenicx_anechoic_annotations: str
    phenicx_anechoic_annotations_tsv: str

    def __init__(self, path='datasets/PHENICX-Anechoic', groups=None, logger_filepath: str = None):
        self.phenicx_anechoic_midi = os.path.join(path, '')
        self.phenicx_anechoic_mixaudio_wav = os.path.join(path, 'mixaudio_wav_22050_mono')
        self.phenicx_anechoic_annotations = os.path.join(path, 'annotations')
        self.phenicx_anechoic_annotations_tsv = os.path.join(path, '_ann_audio_note_tsv')

        super().__init__(path, groups, logger_filepath)

    @classmethod
    def available_groups(cls) -> List[str]:
        """
        It would be also possible to implement instrument-based groups
        """
        return ['beethoven', 'bruckner', 'mahler', 'mozart']

    def get_files(self, group: str) -> List[Tuple[str, str]]:
        logger.info(f'Loading files for group {group}, searching in {self.phenicx_anechoic_mixaudio_wav}')

        audio_filepath: str = os.path.join(self.phenicx_anechoic_mixaudio_wav, group + '.wav')
        midi_path: str = os.path.join(self.phenicx_anechoic_annotations, group, 'all.mid')

        return SchubertWinterreiseDataset.create_audio_tsv([(audio_filepath, midi_path)],
                                                           self.phenicx_anechoic_annotations_tsv)

    @staticmethod
    def load_annotations(annotation_path: str) -> np.ndarray:
        return np.loadtxt(annotation_path, delimiter='\t', skiprows=1)

