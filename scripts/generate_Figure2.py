import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.figure_utilities import (GOLDEN_RATIO, TWO_COLUMN, save_figure,
                                  set_figure_defaults)
from src.sorted_spikes_simulation import make_simulated_run_data


def generate_figure():
    set_figure_defaults()

    (time, linear_distance, sampling_frequency,
     spikes, place_fields) = make_simulated_run_data()
    spike_ind, neuron_ind = np.nonzero(spikes)
    cmap = plt.get_cmap('tab20')

    fig, axes = plt.subplots(3, 1, figsize=(
        TWO_COLUMN, TWO_COLUMN * GOLDEN_RATIO), constrained_layout=True)
    for place_field, color in zip(place_fields.T, cmap.colors):
        axes[0].plot(linear_distance, place_field, linewidth=3, color=color)
    axes[0].set_xlabel('Position [cm]')
    axes[0].set_ylabel('Firing Rate\n[spikes / s]')
    axes[0].set_title('Simulated Neuron Place Fields')
    axes[0].set_xlim((linear_distance.min(), linear_distance.max()))
    axes[0].set_yticks([0, np.round(place_fields.max())])
    axes[0].text(-0.1, 1.0, 'a', transform=axes[0].transAxes,
                 size=15, weight='bold')

    axes[1].plot(time, linear_distance, linewidth=3)
    axes[1].set_ylabel('Position [cm]')
    axes[1].set_title('Simulated Position and Spikes')
    axes[1].set_yticks([0, np.round(linear_distance.max())])
    axes[1].set_xticks([])
    axes[1].set_xlim((0.0, 90.0))
    axes[1].text(-0.1, 1.0, 'b', transform=axes[1].transAxes,
                 size=15, weight='bold')

    c = [cmap.colors[ind] for ind in neuron_ind]
    axes[2].scatter(time[spike_ind], neuron_ind + 1, c=c, s=0.5)
    axes[2].set_yticks((1, spikes.shape[1]))
    axes[2].set_ylabel('Cells')

    axes[2].set_xlabel('Time [s]')
    axes[2].set_xlim((0.0, 90.0))

    sns.despine()

    save_figure('Figure2', figure_format='pdf')
    save_figure('Figure2', figure_format='png')


if __name__ == '__main__':
    sys.exit(generate_figure())
