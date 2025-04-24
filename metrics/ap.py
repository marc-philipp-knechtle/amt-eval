from typing import List, Tuple

import scipy.integrate as sc_integrate
from sklearn.metrics import auc

import matplotlib.pyplot as plt


def calc_ap_from_prec_recall_pairs(precision_recall_pairs: List[Tuple[float, float]], plot: bool = False,
                                   thresholds: List[float] = None, title: str = None) -> float:
    """
    Calculate AP with sklearn.metrics.auc (Area Under Curve)
    This is an alternative method to sklearn.metrics.average_precision_score
    We cannot use said method in some cases because we rely on multiple thresholds (e.g. onset thresh and frame thresh)
    Args:
        precision_recall_pairs: List of Tuples with (precision, recall) values.
        plot: ...
        thresholds: ...
        title: ...
    Returns: Average Precision score, calculated with sklearn
    """
    precision_recall_pairs_sorted_precision = sorted(precision_recall_pairs, key=lambda pair: pair[0])
    precision, recall = zip(*precision_recall_pairs_sorted_precision)
    ap = auc(precision, recall)
    if plot:
        plot_rec_rec_curve(precision, recall, thresholds, title)
    assert ap >= 0
    return ap


def calc_ap_from_prec_recall_pairs_manual(precision_recall_pairs: List[Tuple[float, float]]) -> float:
    precision_recall_pairs_sorted_precision = sorted(precision_recall_pairs, key=lambda pair: pair[0])
    precision, recall = zip(*precision_recall_pairs_sorted_precision)
    # Using recall because it converges to 0 when the threshold becomes 1
    ap = sc_integrate.simpson(recall, precision)
    assert ap >= 0
    return ap


def plot_rec_rec_curve(precision: List[float], recall: List[float], thresholds: List[float] = None, title: str = None):
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.plot(recall, precision)
    if thresholds is not None:
        thresholds = tuple(thresholds)
        for x, y, thr in zip(recall, precision, thresholds):
            ax.annotate(f"({thr:.2f})", (x, y), textcoords="offset points", xytext=(5, 5), ha='center')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title if title is not None else 'Recall-Recall Curve')
    plt.show()
