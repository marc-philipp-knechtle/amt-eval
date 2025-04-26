import os

from data.dataset import PhenicxAnechoicDataset, SchubertWinterreiseDataset, WagnerRingDataset, Bach10Dataset, \
    RwcDataset, TriosDataset, ChoralSingingDataset, MusicNetDataset

tests_root = os.path.abspath('/media/mpk/external-nvme/amt-eval')
"""
used so that the directories etc. can be referenced without using root path 
"""

dataset_definitions_minimal = {
    'SWD': lambda: SchubertWinterreiseDataset(path='tests/dataset_fixtures/Schubert_Winterreise_Dataset_v2-1', groups=['HU33'], neither_split='test'),
    'WRD': lambda: WagnerRingDataset(path='tests/dataset_fixtures/WagnerRing_v0-1', groups=['Furtwangler1953']),
    'B10': lambda: Bach10Dataset(path='tests/dataset_fixtures/Bach10', groups=['07']),
    'PhA': lambda: PhenicxAnechoicDataset(path='tests/dataset_fixtures/PHENICX-Anechoic', groups= ['beethoven']),
    'RWC': lambda: RwcDataset(path='tests/dataset_fixtures/RWC', groups=['rwc']),
    'Trios': lambda: TriosDataset(path='tests/dataset_fixtures/TRIOS', groups=['brahms']),
    'CSD': lambda: ChoralSingingDataset(path='tests/dataset_fixtures/ChoralSingingDataset', groups=['Bruckner_LocusIste']),
    'MuN': lambda: MusicNetDataset(path='tests/dataset_fixtures/MusicNet', groups=['MuN-3-test']),
}


def test_build_all_datasets():
    for key, value in dataset_definitions_minimal.items():
        dataset = value()
        assert str(dataset) is not None # -> just load the dataset with a testing group
