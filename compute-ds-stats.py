import mido
from tqdm import tqdm

from data import dataset_determination


def main():
    dataset = dataset_determination.dataset_definitions_trans_comparing_paper['B10']()

    for label in tqdm(dataset):
        midipath = label[1]
        midi = mido.MidiFile(midipath)
        instruments = set()

        for track in midi.tracks:
            for msg in track:
                if msg.type == 'program_change':
                    instruments.add(msg.program)

        print(f'Instruments in {midipath}: {instruments}')

if __name__ == '__main__':
    main()
