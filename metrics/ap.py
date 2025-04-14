from typing import List, Tuple

from sklearn.metrics import auc


def calc_ap_from_prec_recall_pairs(precision_recall_pairs: List[Tuple[float, float]]) -> float:
    precision_recall_pairs_sorted_precision = sorted(precision_recall_pairs, key=lambda pair: pair[0])
    precision, recall = zip(*precision_recall_pairs_sorted_precision)
    ap = auc(precision, recall)
    assert ap >= 0
    return ap

def calc_ap_from_prec_recall_pairs_manual(precision_recall_pairs: List[Tuple[float, float]]) -> float:
    precision_recall_pairs = sorted(precision_recall_pairs)
    total_precision_recall_area = 0
    prev_recall = 0
    prev_precision = 0
    for precision, recall in precision_recall_pairs:
        area_to_add = (precision - prev_precision) * max(recall, prev_recall)
        assert area_to_add >= 0
        total_precision_recall_area += area_to_add
        prev_precision = precision
        prev_recall = recall

    assert total_precision_recall_area >= 0
    return total_precision_recall_area
