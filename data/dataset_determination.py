from pathlib import Path

from data.dataset import Bach10Dataset, ChoralSingingDataset, MusicNetDataset


def dir_contains_other_dirs(dir_path) -> bool:
    path = Path(dir_path)
    return any(entry.is_dir() for entry in path.iterdir())

dataset_definitions_trans_comparing_paper = {
    'B10': lambda: Bach10Dataset(groups=['07', '08', '09', '10']),
    'CSD': lambda: ChoralSingingDataset(groups=['Bruckner_LocusIste']),
    'MuN': lambda: MusicNetDataset(groups=['MuN-10-var-test']),
    'PhA': lambda: Bach10Dataset(groups=['bruckner', 'mozart']),
    'RWC': lambda: Bach10Dataset(groups=['07', '08', '09', '10']),
    'SWD': lambda: Bach10Dataset(groups=['07', '08', '09', '10']),
    'Trios': lambda: Bach10Dataset(groups=['07', '08', '09', '10']),
}