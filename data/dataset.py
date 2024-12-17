import logging
import os.path
import re
from abc import abstractmethod
from glob import glob
from typing import List, Dict, Tuple

import librosa
import numpy as np
from torch.utils.data import Dataset
from torch import Tensor
import soundfile
import torch

from constants import SAMPLE_RATE, MAX_MIDI, MIN_MIDI, HOP_LENGTH, HOPS_IN_ONSET, HOPS_IN_OFFSET
from utils import midi


class NoteTrackingDataset(Dataset):
    path: str
    groups: List[str]
    # sequence_length: int
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data: List[Dict[str, Tensor]]
    """
    basically the .pt files which are saved automatically. 
    """

    def __init__(self, path: str, groups: List[str] = None):
        self.path = path
        self.groups = groups if groups is not None else self.available_groups()

        print(f"Loading {len(self.groups)} group{'s' if len(self.groups) > 1 else ''} "
              f"of {self.__class__.__name__} at {self.path}")

        for group in self.groups:
            input_file: Tuple[str, str]
            """
            [str, str] = ['path to audio', 'path to annotation']
            """
            for input_file in self.get_files(group):
                self.data.append(self.load(input_file[0], input_file[1]))

    def __getitem__(self, index):
        data = self.data[index]
        result = dict(path=data['path'])

        # if self.sequence_length is not None:
        #     audio_length = len(data['audio'])
        #     step_begin = self.random.randint(audio_length - self.sequence_length) // HOP_LENGTH
        #     n_steps = self.sequence_length // HOP_LENGTH
        #     step_end = step_begin + n_steps
        #
        #     begin = step_begin * HOP_LENGTH
        #     end = begin + self.sequence_length
        #
        #     result['audio'] = data['audio'][begin:end].to(self.device)
        #     result['label'] = data['label'][step_begin:step_end, :].to(self.device)
        #     result['velocity'] = data['velocity'][step_begin:step_end, :].to(self.device)
        # else:
        result['audio'] = data['audio'].to(self.device)
        result['label'] = data['label'].to(self.device)
        result['velocity'] = data['velocity'].to(self.device).float()

        result['audio'] = result['audio'].float().div_(32768.0)
        result['onset'] = (result['label'] == 3).float()
        result['offset'] = (result['label'] == 1).float()
        result['frame'] = (result['label'] > 1).float()
        result['velocity'] = result['velocity'].float().div_(128.0)

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
        :param group:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def load_annotations(self, annotation_path: str) -> np.ndarray:
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
        audio, sr = soundfile.read(audio_path, dtype='int64', always_2d=True)
        # Conversion to float see:
        # https://stackoverflow.com/questions/58810035/converting-audio-files-between-pydub-and-librosa
        audio: np.ndarray = np.array(audio).astype(np.float32)
        # converting from 2D representations (left and right channel) to 1D
        audio = audio.T
        audio = audio[0]

        if sr != SAMPLE_RATE:
            logging.info(f"Sample Rate Mismatch: resampling file: {audio_path}, from {sr} to default: {SAMPLE_RATE}")
            audio = librosa.resample(audio, sr, SAMPLE_RATE)

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

    def __init__(self, path='datasets/Schubert_Winterreise_Dataset_v2-1', groups=None):
        # adding underscore to symbolize that these annotations are computationally created
        self.swd_midi = os.path.join(path, '02_Annotations', '_ann_audio_note_midi')
        self.swd_csv = os.path.join(path, '02_Annotations', 'ann_audio_note')
        self.swd_tsv = os.path.join(path, '02_Annotations', '_ann_audio_note_tsv')
        self.swd_audio_wav = os.path.join(path, '01_RawData', 'audio_wav')

        super().__init__(path, groups)

    def __str__(self):
        return 'SchubertWinterreiseDataset'

    @staticmethod
    def get_filepaths_for_group(directory: str, group_pattern) -> List[str]:
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
        audio_filepaths: List[str] = self.get_filepaths_for_group(self.swd_audio_wav, group)
        if len(audio_filepaths) == 0:
            raise RuntimeError(f'Expected files for group {group}, found nothing.')

        ann_audio_note_filepaths_csv: List[str] = glob(os.path.join(self.swd_csv, '*.csv'))

        # save csv as midi
        midi_path = midi.save_csv_as_midi(ann_audio_note_filepaths_csv, self.swd_midi)
        midi_audio_filepaths: List[str] = glob(os.path.join(midi_path, '*.mid'))
        files_audio_audio_midi: List[Tuple[str, str]] = self.combine_audio_midi(audio_filepaths, midi_audio_filepaths)
        return files_audio_audio_midi


    def load_annotations(self, annotation_path: str) -> np.ndarray:
        pass

    @staticmethod
    def combine_audio_midi(audio_filenames: List[str], midi_filenames: List[str]) -> List[Tuple[str, str]]:
        """
        Args:
            audio_filenames: List of all audio filenames
            midi_filenames: List of all midi filenames
        Returns: audio - midi filename combination in the form of a List of tuples
        """
        audio_midi_combination: List[Tuple[str, str]] = []
        for audio_filename in audio_filenames:
            basename = os.path.basename(audio_filename)
            number_str: str = basename[14:16]
            performance: str = basename[17:21]
            # Find matching midi file
            matching_files = [midi_file for midi_file in midi_filenames if
                              re.compile(fr".*-{number_str}_{performance}.*").search(midi_file)]
            if len(matching_files) > 1:
                raise RuntimeError(f"Found more than one matching file for audio filename: {audio_filename}")
            midi_filepath: str = matching_files[0]
            # Create tuple
            audio_midi_combination.append((os.path.basename(audio_filename), os.path.basename(midi_filepath)))
        return audio_midi_combination