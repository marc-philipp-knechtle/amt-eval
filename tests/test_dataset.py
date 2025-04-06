import os

from data.dataset import PhenicxAnechoicDataset, SchubertWinterreiseDataset, WagnerRingDataset, Bach10Dataset, \
    RwcDataset, TriosDataset, ChoralSingingDataset, MusicNetDataset

tests_root = os.path.abspath(os.path.dirname(__file__))
"""
used so that the directories etc. can be referenced without using root path 
"""

dataset_definitions_minimal = {
    'SWD': lambda: SchubertWinterreiseDataset(path='dataset_fixtures/Schubert_Winterreise_Dataset_v2-1', groups=['HU33'], neither_split='test'),
    'WRD': lambda: WagnerRingDataset(path='dataset_fixtures/WagnerRing_v0-1', groups=['Furtwangler1953']),
    'B10': lambda: Bach10Dataset(path='dataset_fixtures/Bach10', groups=['07']),
    'PhA': lambda: PhenicxAnechoicDataset(path='dataset_fixtures/PHENICX-Anechoic', groups= ['beethoven']),
    'RWC': lambda: RwcDataset(groups=['rwc']),
    'Trios': lambda: TriosDataset(groups=['brahms']),
    'CSD': lambda: ChoralSingingDataset(groups=['Bruckner_LocusIste']),
    'MuN': lambda: MusicNetDataset(groups=['MuN-10-var-test']),
}


def test_build_all_datasets():
    for key, value in dataset_definitions_minimal.items():
        dataset = value()
        assert str(dataset) is not None # -> just load the dataset with a testing group
