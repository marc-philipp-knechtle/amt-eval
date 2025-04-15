from metrics.ap import calc_ap_from_prec_recall_pairs, calc_ap_from_prec_recall_pairs_manual


def test_ap_zero():
    prec_recall_values0 = [(0.0, 0.0), (0.0, 0.0)]
    assert calc_ap_from_prec_recall_pairs(prec_recall_values0) == 0


def test_ap_one():
    prec_recall_values1 = [(1.0, 1.0), (1.0, 1.0)]
    assert calc_ap_from_prec_recall_pairs(prec_recall_values1) == 1


def test_ap_integral():
    # imaginary thresholds: [0, 1]
    prec_recall_values = [(0.0, 1.0), (1.0, 0.0)]
    thresholds = [0.0, 1.0]
    ap = calc_ap_from_prec_recall_pairs(prec_recall_values, True, thresholds)

    assert ap == 0.5


def test_ap_mixed():
    prec_recall_values_sorted = [(1.0, 0.0), (0.7, 0.2), (0.5, 0.35), (0.4, 0.7), (0.0, 1.0)]
    prec_recall_values_mixed = [(0.7, 0.2), (0.0, 1.0), (0.5, 0.35), (0.4, 0.7), (1.0, 0.0)]

    assert calc_ap_from_prec_recall_pairs(prec_recall_values_sorted) == calc_ap_from_prec_recall_pairs(
        prec_recall_values_mixed)
