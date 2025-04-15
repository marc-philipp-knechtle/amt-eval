from typing import List, Tuple

import scipy.integrate as sc_integrate
from sklearn.metrics import auc


def calc_ap_from_prec_recall_pairs(precision_recall_pairs: List[Tuple[float, float]]) -> float:
    precision_recall_pairs_sorted_precision = sorted(precision_recall_pairs, key=lambda pair: pair[0])
    precision, recall = zip(*precision_recall_pairs_sorted_precision)
    ap = auc(precision, recall)
    assert ap >= 0
    return ap


def calc_ap_from_prec_recall_pairs_manual(precision_recall_pairs: List[Tuple[float, float]]) -> float:
    precision_recall_pairs_sorted_precision = sorted(precision_recall_pairs, key=lambda pair: pair[0])
    precision, recall = zip(*precision_recall_pairs_sorted_precision)
    # Using recall because it converges to 0 when the threshold becomes 1
    ap = sc_integrate.simpson(recall, precision)
    assert ap >= 0
    return ap
