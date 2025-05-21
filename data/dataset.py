import logging
import os.path
import re
import sys
from abc import abstractmethod
from glob import glob
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import pretty_midi
from torch.utils.data import Dataset
import torch

from utils import midi, log

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = log.CustomFormatter()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class AmtEvalDataset(Dataset):
    data: List[Tuple[str, str]]

    def __init__(self, data: List[Tuple[str, str]]):
        self.data = data

    def __iter__(self):
        for i in range(len(self.data)):
            yield self[i]


class NoteTrackingDataset(AmtEvalDataset):
    path: str
    groups: List[str]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data: List[Dict[str, str]]

    def __init__(self, path: str, groups: List[str] = None, logging_filepath: str = None):
        self.path = path
        self.groups = groups if groups is not None else self.available_groups()
        self.data: List[Tuple[str, str]] = []
        """
        data for this dataset
        List of Dict[audio_filepath, midi_filepath]
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
            [str, str] = ['path to audio', 'path to tsv annotation']
            """
            for input_file in self.get_files(group):
                self.data.append((input_file[0], input_file[1]))

        super().__init__(self.data)

    def __getitem__(self, index: int) -> Tuple[str, str]:
        return self.data[index]

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


class SchubertWinterreiseDataset(NoteTrackingDataset):
    swd_midi_path: str
    swd_csv_path: str
    swd_audio_wav_path: str
    neither_split: str

    def __init__(self, path='datasets/Schubert_Winterreise_Dataset_v2-1', groups=None, logger_filepath: str = None,
                 neither_split=None):
        """
        :param neither_split: Implements the neither split like specified in the comparing paper:
            Options train, validation, test
            testing: HU33, SC06, 17-24
            validation: AL98, FI55 14-16
            train: FI66, FI80, OL06, QU98, TR99 1-13
        :return: the instance lol
        """
        # adding underscore to symbolize that these annotations are computationally created
        self.swd_midi_path = os.path.join(path, '02_Annotations', '_ann_audio_note_midi')
        self.swd_csv_path = os.path.join(path, '02_Annotations', 'ann_audio_note')
        self.swd_audio_wav_path = os.path.join(path, '01_RawData', 'audio_wav')

        self.neither_split = neither_split

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
        audio_filepaths: List[str] = sorted(self.get_filepaths_for_group(self.swd_audio_wav_path, group))
        if len(audio_filepaths) == 0:
            raise RuntimeError(f'Expected files for group {group}, found nothing.')

        if self.neither_split is not None:
            if self.neither_split == 'train':
                audio_filepaths = audio_filepaths[:13]
            elif self.neither_split == 'validation':
                audio_filepaths = audio_filepaths[13:16]
            elif self.neither_split == 'test':
                audio_filepaths = audio_filepaths[16:25]

        ann_audio_note_filepaths_csv: List[str] = glob(os.path.join(self.swd_csv_path, '*.csv'))
        assert len(ann_audio_note_filepaths_csv) > 0

        # save csv as midi
        midi_path = midi.save_csv_as_midi(ann_audio_note_filepaths_csv, self.swd_midi_path)
        midi_filepaths: List[str] = glob(os.path.join(midi_path, '*.mid'))

        # combine .wav with .midi
        filepaths_audio_midi: List[Tuple[str, str]] = self._combine_audio_midi(audio_filepaths, midi_filepaths)

        # return self.create_audio_tsv(filepaths_audio_midi, self.swd_tsv)
        return filepaths_audio_midi

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
        :return: List of audio filepath with midi filepath
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

        return filepaths_audio_midi

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
    bach10_audio_wav: str

    def __init__(self, path='datasets/Bach10', groups=None, logger_filepath: str = None):
        # underscore = directories are computationally created and can be deleted without worrying
        self.bach10_midi = os.path.join(path, '_ann_audio_note_midi')
        self.bach10_csv = os.path.join(path, 'ann_audio_pitch_CSV')
        self.bach10_audio_wav = os.path.join(path, 'audio_wav_44100_mono')

        super().__init__(path, groups, logger_filepath)

    def __str__(self):
        return 'Bach10Dataset'

    @classmethod
    def available_groups(cls) -> List[str]:
        """
        :return: just one group because Bach10 has no different groups.
        """
        return ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

    def get_files(self, group: str) -> List[Tuple[str, str]]:
        """
        :return: List of Tuple[audio_filename.wav, midi_filename.wav]
        """
        logger.info(f"Loading Files for group {group}, searching in {self.bach10_audio_wav}")

        audio_filepaths: List[str] = glob(os.path.join(self.bach10_audio_wav, group + '*' + '*.wav'))
        if len(audio_filepaths) != 1:
            raise RuntimeError(f'Expected one file for group {group}, found {len(audio_filepaths)}.')

        ann_audio_note_filepaths_csv: List[str] = glob(os.path.join(self.bach10_csv, group + '*'))
        if len(ann_audio_note_filepaths_csv) != 1:
            raise RuntimeError(
                f'Expected one annotatiimon file for group {group}, found {len(ann_audio_note_filepaths_csv)}.')

        # save csv as midi
        midi_path = midi.save_nt_csv_as_midi(ann_audio_note_filepaths_csv, self.bach10_midi)
        midi_filepaths: List[str] = glob(os.path.join(midi_path, '*.mid'))

        # combine .wav with .mid
        filepaths_audio_midi: List[Tuple[str, str]] = WagnerRingDataset._combine_audio_midi(audio_filepaths,
                                                                                            midi_filepaths)
        return filepaths_audio_midi


class PhenicxAnechoicDataset(NoteTrackingDataset):
    """
    This implementation currently does not consider the option of having single instruments as groups.
    """
    phenicx_anechoic_mixaudio_wav: str
    phenicx_anechoic_annotations: str

    def __init__(self, path='datasets/PHENICX-Anechoic', groups=None, logger_filepath: str = None):
        self.phenicx_anechoic_mixaudio_wav = os.path.join(path, 'mixaudio_wav_22050_mono')
        self.phenicx_anechoic_annotations = os.path.join(path, 'annotations')

        super().__init__(path, groups, logger_filepath)

    def __str__(self):
        return 'PhenicxAnechoicDataset'

    @classmethod
    def available_groups(cls) -> List[str]:
        """
        It would be also possible to implement instrument-based groups
        """
        return ['beethoven', 'bruckner', 'mahler', 'mozart']

    def get_files(self, group: str) -> List[Tuple[str, str]]:
        logger.info(f'Loading files for group {group}, searching in {self.phenicx_anechoic_mixaudio_wav}')

        audio_filepath: str = os.path.join(self.phenicx_anechoic_mixaudio_wav, group + '.wav')

        midi_filepaths: List[str] = glob(os.path.join(self.phenicx_anechoic_annotations, group, '*.mid'))
        # remove the all.mid file, where all the _o files are included
        midi_filepaths = [f for f in midi_filepaths if not re.compile(fr".*all.mid").search(f)]
        # remove all original files (not warped to the actual recording)
        midi_filepaths = [f for f in midi_filepaths if not re.compile(fr".*_o.mid").search(f)]

        midi_path: str = midi.combine_midi_files(midi_filepaths, os.path.join(self.phenicx_anechoic_annotations, group,
                                                                              'warped_all.mid'))

        # For this implementation, we only have one file per group -> this is enough
        return [(audio_filepath, midi_path)]

    @staticmethod
    def load_annotations(annotation_path: str) -> np.ndarray:
        return np.loadtxt(annotation_path, delimiter='\t', skiprows=1)


class RwcDataset(NoteTrackingDataset):
    rwc_wav: str
    rwc_midi_warped: str

    def __init__(self, path='datasets/RWC', groups=None, logger_filepath: str = None):
        self.rwc_wav = os.path.join(path, 'wav_22050_mono')
        self.rwc_midi_warped = os.path.join(path, 'MIDI_warped')

        super().__init__(path, groups, logger_filepath)

    def __str__(self):
        return 'RwcDataset'

    @classmethod
    def available_groups(cls) -> List[str]:
        return ['rwc']

    def get_files(self, group: str) -> List[Tuple[str, str]]:
        logger.info(f'Loading files for group {group}, searching in {self.rwc_wav}')
        audio_filepaths: List[str] = glob(os.path.join(self.rwc_wav, '*.wav'), recursive=False)
        if len(audio_filepaths) == 0:
            raise RuntimeError(f'Expected files for group {group}, found nothing.')

        midi_filepaths: List[str] = glob(os.path.join(self.rwc_midi_warped, '*.mid'), recursive=False)

        # combine .wav with .mid
        filepaths_audio_midi: List[Tuple[str, str]] = WagnerRingDataset._combine_audio_midi(audio_filepaths,
                                                                                            midi_filepaths)
        return filepaths_audio_midi

    @staticmethod
    def load_annotations(annotation_path: str) -> np.ndarray:
        return np.loadtxt(annotation_path, delimiter='\t', skiprows=1)


class TriosDataset(NoteTrackingDataset):
    trios_wav: str
    trios_midi_combined: str

    def __init__(self, path='datasets/TRIOS', groups=None, logger_filepath: str = None):
        self.trios_wav = os.path.join(path, 'mix')
        self.trios_midi_combined = os.path.join(path, '_mix_midi')
        super().__init__(path, groups, logger_filepath)

    def __str__(self):
        return 'TriosDataset'

    @classmethod
    def available_groups(cls) -> List[str]:
        return ['brahms', 'lussier', 'mozart', 'schubert', 'take_five']

    def get_files(self, group: str) -> List[Tuple[str, str]]:
        logger.info(f'Loading files for group {group}, searching in {self.trios_wav}')
        audio_filepath: str = os.path.join(self.trios_wav, group + '.wav')
        if len(audio_filepath) == 0:
            raise RuntimeError(f'Expected files for group {group}, found nothing.')

        midi_parts_filepaths: List[str] = glob(os.path.join(self.path, group, '*.mid'), recursive=False)

        """
         in the mozart and lussier Trio, there are files with a different 
         ticks per beat value.
         This leads to errors when combining those files. Therefore we submit the default value here and resample every 
        """
        midi_filepath: str = midi.combine_midi_files(midi_parts_filepaths,
                                                     os.path.join(self.trios_midi_combined, group + '.mid'),
                                                     default_ticks_per_beat=480)
        return [(audio_filepath, midi_filepath)]

    @staticmethod
    def load_annotations(annotation_path: str) -> np.ndarray:
        return np.loadtxt(annotation_path, delimiter='\t', skiprows=1)


class ChoralSingingDataset(NoteTrackingDataset):
    csd_audio_dir: str
    csd_midi_mixed: str

    def __init__(self, path='datasets/ChoralSingingDataset', groups=None, logger_filepath: str = None):
        self.csd_audio_dir = os.path.join(path, 'mixaudio_wav_22050_mono')
        self.csd_midi_mixed = os.path.join(path, '_ann_audio_note_midi')
        super().__init__(path, groups, logger_filepath)

    def __str__(self):
        return 'ChoralSingingDataset'

    @classmethod
    def available_groups(cls) -> List[str]:
        return ['Bruckner_LocusIste', 'Guerrero_NinoDios', 'Traditional_ElRossinyol']

    def get_files(self, group: str) -> List[Tuple[str, str]]:
        logging.info(f'Loading files for group {group}, searching in {self.path}')
        audio_filepaths: List[str] = glob(os.path.join(self.csd_audio_dir, '*' + group + '*.wav'))
        if len(audio_filepaths) != 5:
            raise RuntimeError(f'Expected exactly 5 files for group {group}, found {len(audio_filepaths)} files.')

        midi_filepaths: List[str] = glob(
            os.path.join(self.path, 'ChoralSingingDataset', 'CSD_' + group, 'midi', '*.mid'), recursive=False)
        if len(midi_filepaths) != 4:
            raise RuntimeError(f'Expected four midi files for group {group}, found {len(midi_filepaths)} files.')

        midi_sorted: Dict = {}
        for midifile in midi_filepaths:
            if 'alt' in midifile:
                midi_sorted['alt'] = midifile
            elif 'sop' in midifile:
                midi_sorted['sop'] = midifile
            elif 'ten' in midifile:
                midi_sorted['ten'] = midifile
            elif 'bas' in midifile:
                midi_sorted['bas'] = midifile
            else:
                raise RuntimeError()

        filepaths_audio_midi: List[Tuple[str, str]] = []
        for audio_file in audio_filepaths:
            if 'alt' in audio_file:
                noalt_midi = midi.combine_midi_files([midi_sorted['sop'], midi_sorted['ten'], midi_sorted['bas']],
                                                     os.path.join(self.csd_midi_mixed, group + 'noalt.mid'))
                filepaths_audio_midi.append((audio_file, noalt_midi))
            elif 'sop' in audio_file:
                nosop_midi = midi.combine_midi_files([midi_sorted['alt'], midi_sorted['ten'], midi_sorted['bas']],
                                                     os.path.join(self.csd_midi_mixed, group + 'nosop.mid'))
                filepaths_audio_midi.append((audio_file, nosop_midi))
            elif 'ten' in audio_file:
                noten_midi = midi.combine_midi_files([midi_sorted['sop'], midi_sorted['alt'], midi_sorted['bas']],
                                                     os.path.join(self.csd_midi_mixed, group + 'noten.mid'))
                filepaths_audio_midi.append((audio_file, noten_midi))
            elif 'bas' in audio_file:
                nobas_midi = midi.combine_midi_files([midi_sorted['sop'], midi_sorted['alt'], midi_sorted['ten']],
                                                     os.path.join(self.csd_midi_mixed, group + 'nobas.mid'))
                filepaths_audio_midi.append((audio_file, nobas_midi))
            else:
                all_midi = midi.combine_midi_files(
                    [midi_sorted['sop'], midi_sorted['alt'], midi_sorted['ten'], midi_sorted['bas']],
                    os.path.join(self.csd_midi_mixed, group + 'all.mid'))
                filepaths_audio_midi.append((audio_file, all_midi))
        return filepaths_audio_midi

    @staticmethod
    def load_annotations(annotation_path: str) -> np.ndarray:
        return np.loadtxt(annotation_path, delimiter='\t', skiprows=1)


class MusicNetDataset(NoteTrackingDataset):
    mun_audio: str
    mun_generated_midi_annotations: str

    MUN_ANNOTATION_SAMPLERATE: int = 44100

    test_set_files: Dict = {
        'MuN-3-test': ['2303', '1819', '2382'],
        'MuN-10-test': ['2303', '1819', '2382', '2298', '2191', '2556', '2416', '2628', '1759', '2106'],
        'MuN-10-var-test': ['2303', '1819', '2382', '2298', '2191', '2556', '2416', '2629', '1759', '2106'],
        'MuN-10-slow-test': ['2302', '1818', '2383', '2293', '2186', '2557', '2415', '2627', '1758', '2105'],
        'MuN-10-fast-test': ['2310', '1817', '2381', '2296', '2186', '2555', '2417', '2626', '1757', '2104'],
        'MuN-36-cyc-test': ['2302', '2303', '2304', '2305',
                            '1817', '1818', '1819',
                            '2381', '2382', '2383', '2384',
                            '2293', '2294', '2295', '2296', '2297', '2298',
                            '2186', '2191',
                            '2555', '2556', '2557',
                            '2415', '2416', '2417',
                            '2626', '2627', '2628', '2629',
                            '1757', '1758', '1759', '1760',
                            '2104', '2105', '2106']
    }

    validation_set_files = ['1729', '1733', '1755', '1756', '1765', '1766', '1805', '1807', '1811', '1828', '1829',
                            '1932', '1933', '2081', '2082', '2083', '2157', '2158', '2167', '2186', '2194', '2221',
                            '2222', '2289', '2315', '2318', '2341', '2342', '2480', '2481', '2629', '2632', '2633']

    def __init__(self, path='datasets/MusicNet', groups=None):
        self.mun_audio = os.path.join(path, 'musicnet')
        self.mun_generated_midi_annotations = os.path.join(path, '_musicnet_generated_midi')

        super().__init__(path, groups)

    def __str__(self):
        return 'MusicNetDataset'

    @staticmethod
    def load_annotations(annotation_path: str) -> np.ndarray:
        return np.loadtxt(annotation_path, delimiter='\t', skiprows=1)

    @classmethod
    def available_groups(cls):
        return ['MuN-3-train', 'MuN-3-test', 'MuN-10-train', 'MuN-10-test', 'MuN-10-var-train', 'MuN-10-var-test',
                'MuN-10-slow-train', 'MuN-10-slow-test', 'MuN-10-fast-train', 'MuN-10-fast-test',
                'MuN-36-cyc-train', 'MuN-36-cyc-test']

    def save_mun_csv_as_midi(self, csv_file, midi_path) -> str:
        if not os.path.exists(midi_path):
            os.mkdir(midi_path)

        csv_annotations: pd.DataFrame = pd.read_csv(csv_file, sep=',')
        midi_filename = os.path.basename(csv_file.replace('.csv', '.mid'))
        midi_filepath = os.path.join(midi_path, midi_filename)
        if os.path.exists(midi_filepath):
            return str(midi_filepath)

        piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
        piano = pretty_midi.Instrument(program=piano_program)

        for idx, row in csv_annotations.iterrows():
            onset: float = row[0] / self.MUN_ANNOTATION_SAMPLERATE
            offset: float = row[1] / self.MUN_ANNOTATION_SAMPLERATE
            pitch: int = int(row[3])
            note = pretty_midi.Note(start=onset, end=offset, pitch=pitch, velocity=64)
            piano.notes.append(note)
        file: pretty_midi.PrettyMIDI = pretty_midi.PrettyMIDI()
        file.instruments.append(piano)
        file.write(midi_filepath)

        return str(midi_filepath)

    def get_files(self, group):
        logging.info(f'Loading files for group {group}, searching in {self.mun_audio}')
        all_audio_filepaths = glob(os.path.join(self.mun_audio, '**', '*.wav'), recursive=True)
        audio_filepaths_filtered: List[str] = []
        if 'test' in group:
            test_labels: List[str] = self.test_set_files[group]
            for filepath in all_audio_filepaths:
                if any(test_label in filepath for test_label in test_labels):
                    audio_filepaths_filtered.append(filepath)
        if 'train' in group:
            group_test = group[:-5] + 'test'
            test_labels: List[str] = self.test_set_files[group_test]
            for filepath in all_audio_filepaths:
                if not any(test_label in filepath for test_label in test_labels):
                    audio_filepaths_filtered.append(filepath)
        elif 'validation' in group:
            for filepath in all_audio_filepaths:
                if any(validation_label in filepath for validation_label in self.validation_set_files):
                    audio_filepaths_filtered.append(filepath)

        if len(audio_filepaths_filtered) < 2:
            raise RuntimeError(
                f'Received unexpected number of files for group {group}, found {len(audio_filepaths_filtered)}')

        filepaths_audio_midi: List[Tuple[str, str]] = []
        for file in audio_filepaths_filtered:
            identifier = os.path.basename(file)[:-4]
            csv_files = glob(os.path.join(self.mun_audio, '**', identifier + '*.csv'), recursive=True)
            if len(csv_files) != 1:
                raise RuntimeError(f'Expected 1 file for {file}, got {len(csv_files)}')
            midi_filepath = self.save_mun_csv_as_midi(csv_files[0], self.mun_generated_midi_annotations)
            filepaths_audio_midi.append((file, midi_filepath))

        return filepaths_audio_midi
