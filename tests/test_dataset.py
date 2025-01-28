import os

import pytest

from data.dataset import PhenicxAnechoicDataset

tests_root = os.path.abspath(os.path.dirname(__file__))


def test_swd():
    ...


def test_wrd():
    ...


def test_bach10():
    ...


def test_phenicx_anechoic():
    phenicx_anechoic_dataset = PhenicxAnechoicDataset(
        path=os.path.join(tests_root, 'dataset_fixtures/PHENICX-Anechoic/'),
        groups=['beethoven'])
    assert phenicx_anechoic_dataset.groups == ['beethoven']
