import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dask.distributed import Client
from replay_trajectory_classification import SortedSpikesClassifier
from replay_trajectory_classification.state_transition import \
    estimate_movement_var

from src.figure_utilities import (GOLDEN_RATIO, PAGE_HEIGHT, TWO_COLUMN,
                                  save_figure, set_figure_defaults)
from src.parameters import STATE_COLORS, TRANSITION_TO_CATEGORY
from src.sorted_spikes_simulation import (make_hover_continuous_hover_replay,
                                          make_simulated_run_data)


def plot_replay_probability(results, diag, ax):
    results = results.assign_coords(
        state=lambda ds: ds.state.to_index()
        .map(TRANSITION_TO_CATEGORY),
        time=lambda ds: ds.time * 1000
    )
    replay_probability = results.acausal_posterior.sum('position')
    for state, prob in replay_probability.groupby('state'):
        ax.plot(prob.time, prob.values, linewidth=3,
                label=state, color=STATE_COLORS[state])
    ax.set_ylim((-0.01, 1.05))
    ax.set_yticks([0, 1])
    ax.set_xlim((results.time.min(), results.time.max()))
    ax.set_xticks([results.time.min(), results.time.max()])
    if np.allclose(diag, 1 / 3):
        ax.set_title(f'diag = 1/3')
    else:
        ax.set_title(f'diag = {diag}')


def generate_figure():
    logging.basicConfig(level=logging.INFO)
    client = Client(processes=False)
    logging.info(client)
    set_figure_defaults()

    logging.info('Simulating data...')
    (time, linear_distance, sampling_frequency,
     spikes, place_fields) = make_simulated_run_data()

    movement_var = estimate_movement_var(linear_distance, sampling_frequency)

    replay_time, test_spikes = make_hover_continuous_hover_replay()
    diags = ([1 / 3] +
             [0.4, 0.5, 0.6, 0.7, 0.8, 0.9] +
             [(1 - 10 ** -n) for n in range(2, 10)] +
             [1])

    results = []

    for diag in diags:
        classifier = SortedSpikesClassifier(
            movement_var=movement_var,
            replay_speed=120,
            spike_model_penalty=0.5,
            place_bin_size=np.sqrt(movement_var),
            discrete_transition_diag=diag)
        classifier.fit(linear_distance, spikes)
        results.append(classifier.predict(test_spikes, replay_time))

    fig, axes = plt.subplots(
        nrows=4, ncols=4, figsize=(TWO_COLUMN, PAGE_HEIGHT * GOLDEN_RATIO),
        constrained_layout=True, sharex=True, sharey=True)

    for ind, (ax, diag, r) in enumerate(zip(axes.flat, diags, results)):
        plot_replay_probability(r, diag, ax)
    axes[0, 0].set_ylabel('Probability')
    axes[3, 0].set_xlabel('Time [ms]')

    legend_handle, legend_labels = ax.get_legend_handles_labels()
    fig.legend(legend_handle, legend_labels, loc='lower center',
               fancybox=False, shadow=False, ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    sns.despine()

    save_figure('Supplemental_Figure3', figure_format='pdf')
    save_figure('Supplemental_Figure3', figure_format='png')


if __name__ == '__main__':
    sys.exit(generate_figure())
