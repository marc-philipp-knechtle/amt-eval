from pathlib import Path

from data.dataset import Bach10Dataset, ChoralSingingDataset, MusicNetDataset, PhenicxAnechoicDataset, RwcDataset, \
    SchubertWinterreiseDataset


def dir_contains_other_dirs(dir_path) -> bool:
    path = Path(dir_path)
    return any(entry.is_dir() for entry in path.iterdir())

dataset_definitions_trans_comparing_paper = {
    'B10': lambda: Bach10Dataset(groups=['07', '08', '09', '10']),
    'CSD': lambda: ChoralSingingDataset(groups=['Bruckner_LocusIste']),
    'MuN': lambda: MusicNetDataset(groups=['MuN-10-var-test']),
    'PhA': lambda: PhenicxAnechoicDataset(groups=['bruckner', 'mozart']),
    'RWC': lambda: RwcDataset(groups=['rwc']),
    'SWD': lambda: SchubertWinterreiseDataset(groups=['HU33', 'SC06'], neither_split='test'),
    'Trios': lambda: Bach10Dataset(groups=['07', '08', '09', '10']),
}