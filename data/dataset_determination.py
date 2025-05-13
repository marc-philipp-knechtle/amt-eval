from pathlib import Path

from data.dataset import Bach10Dataset, ChoralSingingDataset, MusicNetDataset, PhenicxAnechoicDataset, RwcDataset, \
    SchubertWinterreiseDataset, TriosDataset


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
    'Trios': lambda: TriosDataset(groups=['brahms', 'lussier', 'mozart', 'schubert', 'take_five']),
}

validation_dataset_comparing_paper = {
    'B10': lambda: Bach10Dataset(groups=['05', '06']),
    'CSD': lambda: ChoralSingingDataset(groups=['Guerrero_NinoDios']),
    'MuN': lambda: MusicNetDataset(groups=['MuN-validation']),
    # 'PhA': ..., -> PhA is not included in the validation set
    # 'RWC': lambda: RwcDataset(groups=['rwc']),
    'SWD': lambda: SchubertWinterreiseDataset(groups=['AL98', 'FI55'], neither_split='validation'),
    # 'MAESTRO': lambda: MaestroD ... -> todo implement MAESTRO
    # Trios
}
"""
Commented datasets are not part of the normal validation set 
"""