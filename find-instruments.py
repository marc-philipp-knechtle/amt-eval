import os
from glob import glob
from typing import List

import mido
from tqdm import tqdm


def main():
    midi_filepaths: List[str] = glob(os.path.join('datasets/MusicNet/musicnet_midis/*/', '*.mid'), recursive=True)

    surely_piano_files: List[str] = []
    might_be_non_piano_files: List[str] = []

    for midi_filepath in tqdm(midi_filepaths):
        midi = mido.MidiFile(midi_filepath)
        is_piano_file = False
        # To ensure that there are program_change messages, otherwise, we assume that it's piano solo!
        multiple_programs: bool = False
        if '2304' in midi_filepath:
            print('asdf')
        for msg in midi:
            if msg.type == 'track_name':
                if 'piano' in msg.name.lower():
                    surely_piano_files.append(midi_filepath)
                    is_piano_file = True
                    break
            if msg.type == 'program_change':
                multiple_programs = True
                if 0 <= msg.program <= 6:
                    surely_piano_files.append(midi_filepath)
                    is_piano_file = True
                    break

        if not is_piano_file and multiple_programs:
            might_be_non_piano_files.append(midi_filepath)

    print(str(might_be_non_piano_files))

    composer_id_mapping = {
        'Bach': [],
        'Brahms': [],
        'Haydn': [],
        'Ravel': [],
        'Beethoven': [],
        'Dvorak': [],
        'Schubert': [],
        'Cambini': [],
        'Mozart': []
    }
    idents = []
    for path in might_be_non_piano_files:
        basename = os.path.basename(path)
        composer = os.path.basename(os.path.dirname(path))
        id = basename.split('_')[0]
        idents.append(id)
        composer_id_mapping[composer].append(id)

    print(str(idents))
    latex_table_str = ''
    for composer, ids in composer_id_mapping.items():
        ids = ', '.join(ids)
        tableline: str = f'{composer} & {ids}\n'
        latex_table_str += tableline

    print(latex_table_str)

if __name__ == "__main__":
    main()