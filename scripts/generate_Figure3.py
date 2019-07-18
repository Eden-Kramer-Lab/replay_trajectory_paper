import logging
import string

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from replay_trajectory_classification import SortedSpikesClassifier
from replay_trajectory_classification.state_transition import \
    estimate_movement_var

from src.figure_utilities import TWO_COLUMN, save_figure, set_figure_defaults
from src.parameters import STATE_COLORS, TRANSITION_TO_CATEGORY
from src.sorted_spikes_simulation import (make_continuous_replay,
                                          make_fragmented_continuous_fragmented_replay,
                                          make_fragmented_hover_fragmented_replay,
                                          make_fragmented_replay,
                                          make_hover_continuous_hover_replay,
                                          make_hover_replay,
                                          make_simulated_run_data)

replay_types = [
    ('continuous', make_continuous_replay),
    ('hover', make_hover_replay),
    ('fragmented', make_fragmented_replay),
    ('hover-cont.-hover', make_hover_continuous_hover_replay),
    ('frag.-cont.-frag.', make_fragmented_continuous_fragmented_replay),
    ('frag.-hover-frag.', make_fragmented_hover_fragmented_replay)
]


def plot_classification(test_spikes, results, subplot_spec, fig, replay_name,
                        letter):
    results = results.assign_coords(
        state=lambda ds: ds.state.to_index()
        .map(TRANSITION_TO_CATEGORY),
        time=lambda ds: ds.time * 1000.0,
    )
    spike_time_ind, neuron_ind = np.nonzero(test_spikes)
    replay_time = results.time.values

    inner_grid = gridspec.GridSpecFromSubplotSpec(
        nrows=3, ncols=1, subplot_spec=subplot_spec)

    # Spikes
    ax = plt.Subplot(fig, inner_grid[0])
    ax.scatter(replay_time[spike_time_ind], neuron_ind, color='black',
               zorder=1, marker='|', s=10, linewidth=1)
    ax.set_yticks((0, test_spikes.shape[1]))
    ax.set_xticks([])
    ax.set_xlim((replay_time.min(), replay_time.max()))
    ax.set_ylabel('Neuron Index')
    ax.text(-0.6, 1.0, letter, transform=ax.transAxes,
            size=20, weight='bold')
    ax.set_title(replay_name)
    fig.add_subplot(ax)

    # Probability
    ax = plt.Subplot(fig, inner_grid[1])
    replay_probability = results.acausal_posterior.sum('position')
    for state, prob in replay_probability.groupby('state'):
        ax.plot(prob.time, prob.values, linewidth=2,
                label=state, color=STATE_COLORS[state])
    ax.set_ylabel('Probability')
    ax.set_yticks([0, 1])
    ax.set_xticks([])
    ax.set_xlim((replay_time.min(), replay_time.max()))
    ax.set_ylim((-0.01, 1.05))
    fig.add_subplot(ax)
    legend_handle, legend_labels = ax.get_legend_handles_labels()

    # Posterior
    ax = plt.Subplot(fig, inner_grid[2])
    quad_mesh = results.acausal_posterior.sum('state').plot(
        x='time', y='position', robust=True, vmin=0.0, ax=ax,
        add_colorbar=False)
    ax.set_ylabel('Position [cm]')
    ax.set_xlim((replay_time.min(), replay_time.max()))
    ax.set_xticks((replay_time.min(), replay_time.max()))
    ax.set_yticks((0.0, 180.0))
    ax.set_xlabel('Time [ms]')
    fig.add_subplot(ax)

    sns.despine()

    return legend_handle, legend_labels, quad_mesh


def generate_figure():
    logging.basicConfig(level=logging.INFO)
    set_figure_defaults()

    logging.info('Simulating data...')
    (time, linear_distance, sampling_frequency,
     spikes, place_fields) = make_simulated_run_data()

    movement_var = estimate_movement_var(linear_distance, sampling_frequency)

    classifier = SortedSpikesClassifier(movement_var=movement_var,
                                        replay_speed=120,
                                        spike_model_penalty=0.5,
                                        place_bin_size=np.sqrt(movement_var))
    classifier.fit(linear_distance, spikes)

    # Make Figure
    logging.info('Making figure...')
    fig = plt.figure(figsize=(TWO_COLUMN, TWO_COLUMN * 1.4),
                     constrained_layout=True)
    outer_grid = fig.add_gridspec(nrows=3, ncols=3, height_ratios=[10, 10, 1])
    for replay_ind, (replay_name, make_replay) in enumerate(replay_types):
        replay_time, test_spikes = make_replay()
        results = classifier.predict(test_spikes, time=replay_time)
        letter = string.ascii_lowercase[replay_ind]
        legend_handle, legend_labels, quad_mesh = plot_classification(
            test_spikes, results, outer_grid[replay_ind], fig, replay_name,
            letter)
    legend_ax = fig.add_subplot(outer_grid[-1, :2])
    legend_ax.axis('off')
    legend_ax.legend(legend_handle, legend_labels, loc='upper center',
                     fancybox=False, shadow=False, ncol=3, frameon=False)

    colorbar_ax = fig.add_subplot(outer_grid[-1, -1])
    colorbar_ax.axis('off')
    axins = inset_axes(colorbar_ax, width='50%', height='50%',
                       bbox_transform=colorbar_ax.transAxes, borderpad=0,
                       loc='lower center')
    cbar = plt.colorbar(quad_mesh, cax=axins, orientation='horizontal',
                        extend='max', label='posterior', ticks=[0.0, 0.035])
    cbar.ax.set_xticklabels(['0', 'max'])
    cbar.ax.tick_params(labelsize=10, width=0, length=0)
    sns.despine()

    save_figure('Figure3', figure_format='pdf')
    save_figure('Figure3', figure_format='png')


if __name__ == '__main__':
    generate_figure()
