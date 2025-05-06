from typing import Dict, List, Any, Tuple

import numpy as np
from matplotlib import pyplot as plt


def plot_threshold_optimization(onset_values_for_diagram: Dict[float, List],
                                frame_values_for_diagram: Dict[float, List]):
    thrs_onset = []
    thrs_values_onset = []
    thrs_values_frame = []
    thrs_values_combi = []
    for thr, values in onset_values_for_diagram.items():
        thrs_onset.append(thr)
        thrs_values_onset.append(np.mean(values))
        thrs_values_frame.append(np.mean(frame_values_for_diagram[thr]))
        thrs_values_combi.append(np.mean([np.mean(values), np.mean(frame_values_for_diagram[thr])]))

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.plot(thrs_onset, thrs_values_onset, label='F1 Onset Perf for Threshold')
    plt.plot(thrs_onset, thrs_values_frame, label='F1 Frame Perf for Threshold')
    plt.plot(thrs_onset, thrs_values_combi, label='F1 Onset/Frame Perf for Threshold')

    plt.xlabel('Thresholds')
    plt.ylabel('F1 Performance')

    plt.legend()
    plt.show()
