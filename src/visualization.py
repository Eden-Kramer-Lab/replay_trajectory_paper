import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loren_frank_data_processing.position import (get_position_dataframe,
                                                  make_track_graph)
from loren_frank_data_processing.track_segment_classification import (
    get_track_segments_from_graph, plot_track, project_points_to_segment)
from matplotlib.collections import LineCollection
from matplotlib.colorbar import ColorbarBase, make_axes
from src.analysis import maximum_a_posteriori_estimate
from src.parameters import STATE_COLORS, STATE_ORDER

try:
    from upsetplot import UpSet
except ImportError:
    class Upset:
        pass


def plot_2D_position_with_color_time(time, position, ax=None, cmap='plasma',
                                     alpha=None):
    '''

    Parameters
    ----------
    time : ndarray, shape (n_time,)
    position : ndarray, shape (n_time, 2)
    ax : None or `matplotlib.axes.Axes` instance
    cmap : str
    alpha : None or ndarray, shape (n_time,)

    Returns
    -------
    line : `matplotlib.collections.LineCollection` instance
    ax : `matplotlib.axes.Axes` instance

    '''
    if ax is None:
        ax = plt.gca()
    points = position.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(time.min(), time.max())
    cmap = plt.get_cmap(cmap)
    colors = cmap(norm(time))
    if alpha is not None:
        colors[:, -1] = alpha
    lc = LineCollection(segments, colors=colors, zorder=100)
    lc.set_linewidth(4)
    line = ax.add_collection(lc)

    # Set the values used for colormapping
    cax, _ = make_axes(ax, location='bottom')
    cbar = ColorbarBase(cax, cmap=cmap, norm=norm,
                        spacing='proportional',
                        orientation='horizontal')
    cbar.set_label('time')

    return line, ax


def plot_all_positions(position_info, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.plot(position_info.x_position.values, position_info.y_position.values,
            color='lightgrey', alpha=0.5, label='all positions')


def make_movie(position, posterior_density, position_info, map_position,
               spikes, place_field_max, movie_name='video_name.mp4'):
    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    plot_all_positions(position_info, ax=ax)

    ax.set_xlim(position_info.x_position.min() - 1,
                position_info.x_position.max() + 1)
    ax.set_ylim(position_info.y_position.min() + 1,
                position_info.y_position.max() + 1)
    ax.set_xlabel('x-position')
    ax.set_ylabel('y-position')

    position_dot = plt.scatter([], [], s=80, zorder=102, color='b',
                               label='actual position')
    position_line, = plt.plot([], [], 'b-', linewidth=3)
    map_dot = plt.scatter([], [], s=80, zorder=102, color='r',
                          label='replay position')
    map_line, = plt.plot([], [], 'r-', linewidth=3)
    spikes_dot = plt.scatter([], [], s=40, zorder=104, color='k',
                             label='spikes')
    vmax = np.percentile(posterior_density.values, 99)
    ax.legend()
    posterior_density.isel(time=0).plot(
        x='x_position', y='y_position', vmin=0.0, vmax=vmax,
        ax=ax)
    n_frames = posterior_density.shape[0]

    def _update_plot(time_ind):
        start_ind = max(0, time_ind - 5)
        time_slice = slice(start_ind, time_ind)

        position_dot.set_offsets(position[time_ind])
        position_line.set_data(position[time_slice, 0],
                               position[time_slice, 1])

        map_dot.set_offsets(map_position[time_ind])
        map_line.set_data(map_position[time_slice, 0],
                          map_position[time_slice, 1])

        spikes_dot.set_offsets(place_field_max[spikes[time_ind] > 0])

        im = posterior_density.isel(time=time_ind).plot(
            x='x_position', y='y_position', vmin=0.0, vmax=vmax,
            ax=ax, add_colorbar=False)

        return position_dot, im

    movie = animation.FuncAnimation(fig, _update_plot, frames=n_frames,
                                    interval=50, blit=True)
    if movie_name is not None:
        movie.save(movie_name, writer=writer)

    return fig, movie


def plot_ripple_decode_2D(posterior, ripple_position,
                          ripple_spikes, position, linear_position_order,
                          spike_label='Cells'):
    time = posterior.time.values
    map_estimate = maximum_a_posteriori_estimate(posterior.sum('state'))
    spike_time_ind, neuron_ind = np.nonzero(
        np.asarray(ripple_spikes)[:, linear_position_order])
    n_neurons = ripple_spikes.shape[1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    replay_probability = posterior.sum('position')
    for state, prob in replay_probability.groupby('state'):
        axes[0].plot(prob.time, prob.values, linewidth=3, label=state,
                     color=STATE_COLORS[state])
    axes[0].set_ylim((0, 1))

    twin_ax = axes[0].twinx()
    twin_ax.scatter(time[spike_time_ind], neuron_ind, color='black', zorder=1,
                    marker='|', s=40, linewidth=3)
    twin_ax.set_ylim((-0.5, n_neurons - 0.5))
    twin_ax.set_yticks((1, n_neurons))
    twin_ax.set_ylabel(spike_label)

    box = axes[0].get_position()
    axes[0].set_position([box.x0, box.y0 + box.height * 0.1,
                          box.width, box.height * 0.9])

    axes[0].legend(loc='upper right', bbox_to_anchor=(1.0, -0.05),
                   fancybox=False, shadow=False, ncol=1, frameon=False)
    axes[0].set_ylabel('Probability')
    axes[0].set_xlabel('Time [ms]')
    axes[0].set_xlim((time.min(), time.max()))
    axes[0].set_xticks((time.min(), time.max()))

    position = np.asarray(position)
    axes[1].plot(position[:, 0], position[:, 1],
                 color='lightgrey', alpha=0.4, zorder=0)
    plot_2D_position_with_color_time(
        time, map_estimate, ax=axes[1])
    ripple_position = np.asarray(ripple_position)
    axes[1].scatter(ripple_position[:, 0], ripple_position[:, 1],
                    color='black', s=100, label='actual position')
    posterior.sum(['state', 'time']).plot(
        x='x_position', y='y_position', robust=True, cmap='Purples', alpha=0.5,
        ax=axes[1], add_colorbar=False, zorder=0)
    axes[1].set_xlabel('X-Position [cm]')
    axes[1].set_ylabel('Y-Position [cm]')

    axes[1].legend()


def plot_ripple_decode_1D(posterior, ripple_position, ripple_spikes,
                          linear_position_order, position_info,
                          spike_label='Cells', figsize=(10, 7)):
    ripple_spikes = np.asarray(ripple_spikes)
    spike_time_ind, neuron_ind = np.nonzero(
        ripple_spikes[:, linear_position_order])
    ripple_time = posterior.time.values
    min_time, max_time = ripple_time.min(), ripple_time.max()

    fig, axes = plt.subplots(
        3, 1, figsize=figsize,
        constrained_layout=True, sharex=True)

    axes[0].scatter(ripple_time[spike_time_ind], neuron_ind, color='black',
                    zorder=1, marker='|', s=20, linewidth=1)
    axes[0].set_yticks((1, ripple_spikes.shape[1]))
    axes[0].set_xticks([])
    axes[0].set_xlim((min_time, max_time))
    axes[0].set_ylabel(spike_label)

    replay_probability = posterior.sum('position')
    for state, prob in replay_probability.groupby('state'):
        axes[1].plot(prob.time, prob.values, linewidth=3, label=state,
                     color=STATE_COLORS[state])
    axes[1].set_ylabel('Probability')
    axes[1].set_yticks([0, 1])
    axes[1].set_xticks([])
    axes[1].set_xlim((min_time, max_time))
    axes[1].set_ylim((-0.01, 1.05))
    axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5),
                   fancybox=False, shadow=False, ncol=1, frameon=False)

    posterior.sum('state').plot(
        x='time', y='position', robust=True, vmin=0.0, ax=axes[2])
    axes[2].set_ylabel('Position [cm]')
    axes[2].set_xlim((min_time, max_time))
    axes[2].set_xticks((min_time, max_time))
    axes[-1].set_xlabel('Time [ms]')

    max_df = position_info.groupby('arm_name').linear_position2.max()
    min_df = position_info.groupby('arm_name').linear_position2.min()
    axes[2].set_ylim((0, position_info.linear_position2.max()))
    axes[2].set_yticks((0, position_info.linear_position2.max()))

    for arm_name, max_position in max_df.iteritems():
        axes[2].axhline(max_position, color='lightgrey',
                        linestyle='-', linewidth=1)
        axes[2].text(min_time, max_position - 5, arm_name, color='white',
                     fontsize=8, verticalalignment='top')
    for arm_name, min_position in min_df.iteritems():
        axes[2].axhline(min_position, color='lightgrey',
                        linestyle='-', linewidth=1)
    axes[2].plot(ripple_time, ripple_position, color='white', linestyle='--',
                 linewidth=2, alpha=0.7)

    sns.despine()


def plot_neuron_place_field_2D_1D_position(
        position_info, place_field_max, linear_place_field_max,
        linear_position_order):
    position = np.asarray(position_info.loc[:, ['x_position', 'y_position']])
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    axes[0].plot(position[:, 0], position[:, 1], color='lightgrey', alpha=0.4)
    axes[0].set_ylabel('y-position')
    axes[0].set_xlabel('x-position')
    zipped = zip(linear_place_field_max[linear_position_order],
                 place_field_max[linear_position_order])
    for ind, (linear, place_max) in enumerate(zipped):
        axes[0].scatter(place_max[0], place_max[1], s=300, alpha=0.3)
        axes[0].text(place_max[0], place_max[1], linear_position_order[ind],
                     fontsize=15, horizontalalignment='center',
                     verticalalignment='center')
        axes[1].scatter(ind, linear, s=200, alpha=0.3)
        axes[1].text(ind, linear, linear_position_order[ind], fontsize=15,
                     horizontalalignment='center', verticalalignment='center')
    max_df = (position_info.groupby('arm_name').linear_position2.max()
              .iteritems())
    for arm_name, max_position in max_df:
        axes[1].axhline(max_position, color='lightgrey', zorder=0,
                        linestyle='--')
        axes[1].text(0, max_position - 0.2, arm_name, color='lightgrey',
                     horizontalalignment='left', verticalalignment='top',
                     fontsize=12)
    axes[1].set_ylim((-3.0, position_info.linear_position2.max() + 3.0))
    axes[1].set_ylabel('linear position')
    axes[1].set_xlabel('Neuron ID')


def plot_category_counts(replay_info):
    upset = UpSet(replay_info.set_index(STATE_ORDER),
                  sum_over=False, sort_sets_by=None)
    return upset.plot()


def _plot_category(replay_info, category, kind='strip', ax=None,
                   is_zero_mask=False, is_normalized=False, **kwargs):
    is_col = replay_info.columns.str.endswith(f'_{category}')
    if is_zero_mask:
        zero_mask = np.isclose(replay_info.loc[:, is_col], 0.0)
    else:
        zero_mask = np.zeros_like(replay_info.loc[:, is_col], dtype=np.bool)
    if is_normalized:
        norm = replay_info.left_well_position.values[:, np.newaxis]
    else:
        norm = np.ones_like(
            replay_info.left_well_position.values[:, np.newaxis])
    data = (replay_info.loc[:, is_col].mask(zero_mask)
            .rename(columns=lambda c: c.split('_')[0]))
    data /= norm
    if kind == 'strip':
        sns.stripplot(data=data, order=STATE_ORDER, orient='horizontal',
                      palette=STATE_COLORS, ax=ax, **kwargs)
    elif kind == 'violin':
        sns.violinplot(data=data, order=STATE_ORDER, orient='horizontal',
                       palette=STATE_COLORS, ax=ax, cut=0, **kwargs)
    sns.despine(left=True, ax=ax)


def plot_category_duration(replay_info, kind='strip', ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    _plot_category(replay_info, 'duration', kind=kind, ax=ax,
                   is_zero_mask=True, **kwargs)
    ax.set_xlabel('Duration [s]')


def plot_linear_position_of_animal(replay_info, ax=None):
    if ax is None:
        ax = plt.gca()
    pos = pd.concat(
        [replay_info.loc[replay_info[state]]
         .actual_linear_position2.rename(state)
         for state in STATE_ORDER], axis=1)
    sns.violinplot(data=pos, order=STATE_ORDER, orient='horizontal',
                   palette=STATE_COLORS, cut=0, inner=None, bw=0.05, ax=ax)
    plot_linear_position_markers(replay_info, ax=ax)
    ax.set_xlabel('Linear position of animal during ripple [cm]')
    sns.despine(left=True, ax=ax)


def plot_normalized_linear_position_of_animal(replay_info, ax=None):
    if ax is None:
        ax = plt.gca()
    pos = pd.concat(
        [(replay_info.loc[replay_info[state]]
          .actual_linear_position2.rename(state)) /
         replay_info.loc[replay_info[state]].left_well_position.values
         for state in STATE_ORDER], axis=1)
    sns.violinplot(data=pos, order=STATE_ORDER, orient='horizontal',
                   palette=STATE_COLORS, cut=0, inner=None, bw=0.025, ax=ax)
    plot_linear_position_markers(replay_info, is_normalized=True, jitter=0.01,
                                 ax=ax)
    ax.set_xlabel('Normalized linear position of animal during ripple [a.u]')
    sns.despine(left=True, ax=ax)


def plot_replay_distance_from_actual_position(replay_info, kind='strip',
                                              ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    _plot_category(replay_info, 'replay_distance_from_actual_position',
                   kind=kind, ax=ax, **kwargs)
    ax.set_xlabel('Average distance from animal position [cm]')


def plot_replay_distance_from_center_well(replay_info, kind='strip', ax=None,
                                          **kwargs):
    if ax is None:
        ax = plt.gca()
    _plot_category(replay_info, 'replay_distance_from_center_well',
                   kind=kind, ax=ax, **kwargs)
    ax.set_xlabel('Replay distance from center well [cm]')
    ax.axvline(replay_info.center_well_position.mean(), color='lightgrey',
               zorder=0, linestyle='--', alpha=0.5)
    ax.axvline(replay_info.choice_position.mean(), color='lightgrey', zorder=0,
               linestyle='--', alpha=0.5)


def plot_replay_total_distance(replay_info, kind='strip', ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    _plot_category(replay_info, 'replay_total_distance',
                   kind=kind, ax=ax, **kwargs)
    if kind == 'violin':
        ax.set_xscale('log')
    ax.set_xlabel('Replay total distance travelled [cm]')


def plot_replay_total_displacement(replay_info, kind='strip', ax=None,
                                   **kwargs):
    if ax is None:
        ax = plt.gca()
    _plot_category(replay_info, 'replay_total_displacement',
                   kind=kind, ax=ax, **kwargs)
    ax.set_xlabel('Replay total displacement [cm]')


def plot_replay_linear_position(replay_info, kind='strip', ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    _plot_category(replay_info, 'replay_linear_position',
                   kind=kind, ax=ax, **kwargs)
    plot_linear_position_markers(replay_info, ax)
    ax.set_xlabel('Average replay position [cm]')


def plot_replay_norm_linear_position(replay_info, kind='strip', ax=None,
                                     **kwargs):
    if ax is None:
        ax = plt.gca()
    _plot_category(replay_info, 'replay_linear_position',
                   kind=kind, ax=ax, is_normalized=True, **kwargs)
    plot_linear_position_markers(replay_info, is_normalized=True, jitter=0.01,
                                 ax=ax)
    ax.set_xlabel('Average norm. replay position [a.u]')


def plot_replay_linear_position_vs_distance_from_actual_position(replay_info):
    fig, axes = plt.subplots(
        1, len(STATE_ORDER), constrained_layout=True,
        figsize=(len(STATE_ORDER) * 3, 3), sharex=True, sharey=True)
    for ax, state in zip(axes.flat, STATE_ORDER):
        ax.scatter(
            replay_info[f'{state}_replay_linear_position'],
            replay_info[f'{state}_replay_distance_from_actual_position'],
            color=STATE_COLORS[state], s=10)
        ax.set_title(state)
        ax.set_ylim((0, replay_info.replay_distance_from_center_well.max()))
        plot_linear_position_markers(replay_info, ax=ax, fontsize=8)

    sns.despine(left=True)
    axes[0].set_ylabel('Avg. replay distance\nfrom animal position [cm]')
    axes[axes.size // 2].set_xlabel('Avg. replay position [cm]')


def plot_norm_replay_linear_position_vs_distance_from_actual_position(
        replay_info):
    fig, axes = plt.subplots(1, len(STATE_ORDER), constrained_layout=True,
                             figsize=(len(STATE_ORDER) * 3, 3), sharex=True,
                             sharey=True)

    max_distance = replay_info.choice_position.values + np.abs(
        replay_info.right_well_position.values -
        replay_info.right_arm_start.values)
    for ax, state in zip(axes.flat, STATE_ORDER):
        ax.scatter(
            replay_info[f'{state}_replay_linear_position'] /
            replay_info.left_well_position.values,
            replay_info[f'{state}_replay_distance_from_actual_position'] /
            max_distance, color=STATE_COLORS[state], s=10)
        ax.set_title(state)
        ax.set_ylim((0, 1))
        plot_linear_position_markers(
            replay_info, ax=ax, fontsize=8, is_normalized=True, jitter=0.01)

    sns.despine(left=True)
    axes[0].set_ylabel('Avg. norm. replay distance\nfrom animal position [cm]')
    axes[axes.size // 2].set_xlabel('Avg. norm. replay position [cm]')


def plot_actual_position_vs_replay_position(replay_info, kind='scatter',
                                            vmax=30):
    fig, axes = plt.subplots(1, len(STATE_ORDER), constrained_layout=True,
                             figsize=(len(STATE_ORDER) * 3, 3), sharex=True,
                             sharey=True)
    avg_left_well_position = replay_info.left_well_position.mean()
    extent = (0, avg_left_well_position, 0, avg_left_well_position)

    for ax, state in zip(axes.flat, STATE_ORDER):
        if kind == 'scatter':
            ax.scatter(replay_info['actual_linear_position2'],
                       replay_info[f'{state}_replay_linear_position'],
                       color=STATE_COLORS[state], s=20)
        elif kind == 'hexbin':
            cmap = sns.light_palette(STATE_COLORS[state], as_cmap=True)
            h = ax.hexbin(replay_info['actual_linear_position2'],
                          replay_info[f'{state}_replay_linear_position'],
                          gridsize=10, extent=extent, vmin=0.0, vmax=vmax,
                          cmap=cmap)
            ax.set_ylim((0, avg_left_well_position))
            ax.set_xlim((0, avg_left_well_position))
        elif kind == 'kdeplot':
            cmap = sns.light_palette(STATE_COLORS[state], as_cmap=True)
            temp_df = (replay_info
                       .loc[:, ['actual_linear_position2',
                                f'{state}_replay_linear_position']]
                       .dropna())
            sns.kdeplot(temp_df['actual_linear_position2'],
                        temp_df[f'{state}_replay_linear_position'],
                        clip=extent, vmin=0.0, cmap=cmap, ax=ax,
                        bw=(10, 10), shade=True, gridsize=30,
                        shade_lowest=True, n_levels=10)
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.set_ylim((0, avg_left_well_position))
            ax.set_xlim((0, avg_left_well_position))
        ax.set_title(state)
        plot_linear_position_markers(replay_info, ax=ax, fontsize=8, zorder=10)
        plot_linear_position_markers(replay_info, ax=ax, is_vertical=False,
                                     zorder=10)
    sns.despine(left=True)
    axes[0].set_ylabel('Avg. replay position [cm]')
    axes[axes.size // 2].set_xlabel('Actual position [cm]')

    if (kind == 'hexbin'):
        plt.colorbar(h, ax=axes[-1])


def plot_norm_actual_position_vs_replay_position(replay_info, kind='scatter',
                                                 vmax=30):
    fig, axes = plt.subplots(1, len(STATE_ORDER), constrained_layout=True,
                             figsize=(len(STATE_ORDER) * 3, 3), sharex=True,
                             sharey=True)
    for ax, state in zip(axes.flat, STATE_ORDER):
        temp_df = (replay_info
                   .loc[:, ['actual_linear_position2',
                            f'{state}_replay_linear_position',
                            'left_well_position']]
                   .dropna())
        actual_position = (temp_df.actual_linear_position2 /
                           temp_df.left_well_position.values)
        replay_position = (temp_df[f'{state}_replay_linear_position'] /
                           temp_df.left_well_position.values)
        if kind == 'scatter':
            ax.scatter(actual_position, replay_position,
                       color=STATE_COLORS[state], s=20)
        elif kind == 'hexbin':
            cmap = sns.light_palette(STATE_COLORS[state], as_cmap=True)
            h = ax.hexbin(actual_position, replay_position,
                          gridsize=15, extent=(0, 1, 0, 1),
                          vmin=0.0, vmax=vmax, cmap=cmap)
        elif kind == 'kdeplot':
            cmap = sns.light_palette(STATE_COLORS[state], as_cmap=True)
            sns.kdeplot(actual_position, replay_position,
                        clip=(0, 1, 0, 1), vmin=0.0, cmap=cmap, ax=ax,
                        bw=0.05, shade=True, gridsize=30,
                        shade_lowest=True, n_levels=20)
            ax.set_ylabel('')
            ax.set_xlabel('')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_title(state)
        plot_linear_position_markers(replay_info, ax=ax, fontsize=8,
                                     is_normalized=True, jitter=0.01,
                                     zorder=10)
        plot_linear_position_markers(replay_info, ax=ax, is_vertical=False,
                                     is_normalized=True, jitter=0.01,
                                     zorder=10)
    sns.despine(left=True)
    axes[0].set_ylabel('Avg. norm. replay position [a.u]')
    axes[axes.size // 2].set_xlabel('Actual norm. position [a.u]')

    if (kind == 'hexbin'):
        plt.colorbar(h, ax=axes[-1])


def plot_replay_time(replay_info, kind='strip', ax=None, is_min=True,
                     **kwargs):
    if ax is None:
        ax = plt.gca()
    category = 'min_time' if is_min else 'max_time'
    name = 'Start' if is_min else 'End'
    _plot_category(
        replay_info.select_dtypes(np.number) /
        (replay_info.duration.values[:, np.newaxis] * 1000),
        category, kind=kind, ax=ax, **kwargs)
    ax.set_xlim((-0.05, 1.05))
    ax.set_xlabel(f'{name} time [normalized ripple time]')


def plot_replay_speed(replay_info, kind='strip', ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    _plot_category(replay_info.select_dtypes(np.number) * 10_000,
                   'replay_speed', kind=kind, ax=ax, **kwargs)
    plt.xlabel('Average Speed [m / s]')


def plot_replay_velocity(replay_info, ax=None, relative_to='actual_position',
                         **kwargs):
    if ax is None:
        ax = plt.gca()
    is_col = replay_info.columns.str.endswith(
        f'_replay_velocity_{relative_to}')
    sns.violinplot(data=10_000 * (replay_info.loc[:, is_col]
                                  .rename(columns=lambda c: c.split('_')[0])),
                   order=STATE_ORDER, orient='horizontal',
                   palette=STATE_COLORS, scale='width', ax=ax, **kwargs)
    plt.axvline(0, color='lightgrey', linestyle='--', zorder=0, alpha=0.5)
    sns.despine(left=True, ax=ax)
    ax.set_xlabel(
        f'Average velocity relative to {relative_to.strip("_")} [m / s]')


def plot_max_probability(replay_info, ax=None, kind='strip', **kwargs):
    if ax is None:
        ax = plt.gca()
    _plot_category(replay_info, 'max_probability', kind=kind, ax=ax, **kwargs)
    ax.set_xlabel('Maximum probability of state for each ripple')


def plot_population_rate(replay_info, ax=None):
    if ax is None:
        ax = plt.gca()
    data = pd.concat(
        [replay_info.loc[replay_info[state]].n_total_spikes.rename(state) /
         replay_info.loc[replay_info[state]].duration.values
         for state in STATE_ORDER], axis=1)
    sns.violinplot(data=data, order=STATE_ORDER,
                   orient='horizontal',
                   palette=STATE_COLORS, cut=0)
    plt.xlabel('Population firing rate per ripple [spikes / s]')
    sns.despine(left=True)


def plot_n_unique_spiking(replay_info, ax=None, kind='strip',
                          data_type='cells', **kwargs):
    if ax is None:
        ax = plt.gca()
    data = pd.concat(
        [replay_info.loc[replay_info[state]].n_unique_spiking.rename(state)
         for state in STATE_ORDER], axis=1)
    if kind == 'violin':
        sns.violinplot(data=data, order=STATE_ORDER, orient='horizontal',
                       palette=STATE_COLORS, cut=0, ax=ax)
    elif kind == 'strip':
        sns.stripplot(data=data, order=STATE_ORDER, orient='horizontal',
                      palette=STATE_COLORS, ax=ax)
    sns.despine(left=True, ax=ax)
    ax.set_xlabel(f'Number of {data_type} participating per ripple')
    ax.set_xlim((1, np.nanmax(data.values) + 1))


def plot_linear_position_markers(replay_info, ax=None, is_vertical=True,
                                 is_normalized=False,
                                 horizontalalignment='left',
                                 verticalalignment='top', fontsize=9,
                                 color='lightgrey', linestyle='--', alpha=0.5,
                                 zorder=0, jitter=1):
    if is_normalized:
        COLUMNS = ['center_well_position', 'choice_position',
                   'right_well_position', 'left_well_position',
                   'right_arm_start', 'left_arm_start']
        replay_info = (replay_info.loc[:, COLUMNS] /
                       replay_info.left_well_position.values[:, np.newaxis])
    if ax is None:
        ax = plt.gca()
    if is_vertical:
        _, y_max = ax.get_ylim()
        ax.axvline(replay_info.center_well_position.mean(),
                   color=color, zorder=zorder, linestyle=linestyle,
                   alpha=alpha)
        ax.text(replay_info.center_well_position.mean() + jitter, y_max,
                'center', horizontalalignment=horizontalalignment,
                verticalalignment=verticalalignment, fontsize=fontsize,
                color=color)

        ax.axvline(replay_info.choice_position.mean(),
                   color=color, zorder=zorder, linestyle=linestyle,
                   alpha=alpha)
        ax.axvline(replay_info.right_arm_start.mean(),
                   color=color, zorder=zorder, linestyle=linestyle,
                   alpha=alpha)
        ax.axvline(replay_info.right_well_position.mean(),
                   color=color, zorder=zorder, linestyle=linestyle,
                   alpha=alpha)
        ax.text(replay_info.left_arm_start.mean() + jitter, y_max, 'left',
                horizontalalignment=horizontalalignment,
                verticalalignment=verticalalignment,
                fontsize=fontsize, color=color)

        ax.axvline(replay_info.left_arm_start.mean(),
                   color=color, zorder=zorder, linestyle=linestyle,
                   alpha=alpha)
        ax.axvline(replay_info.left_well_position.mean(),
                   color=color, zorder=zorder, linestyle=linestyle,
                   alpha=alpha)
        ax.text(replay_info.right_arm_start.mean() + jitter, y_max, 'right',
                horizontalalignment=horizontalalignment,
                verticalalignment=verticalalignment,
                fontsize=fontsize, color=color)

    else:
        _, x_max = ax.get_xlim()
        ax.axhline(replay_info.center_well_position.mean(),
                   color=color, zorder=zorder, linestyle=linestyle,
                   alpha=alpha)
        ax.axhline(replay_info.choice_position.mean(),
                   color=color, zorder=zorder, linestyle=linestyle,
                   alpha=alpha)
        ax.axhline(replay_info.right_arm_start.mean(),
                   color=color, zorder=zorder, linestyle=linestyle,
                   alpha=alpha)
        ax.axhline(replay_info.right_well_position.mean(),
                   color=color, zorder=zorder, linestyle=linestyle,
                   alpha=alpha)
        ax.axhline(replay_info.left_arm_start.mean(),
                   color=color, zorder=zorder, linestyle=linestyle,
                   alpha=alpha)
        ax.axhline(replay_info.left_well_position.mean(),
                   color=color, zorder=zorder, linestyle=linestyle,
                   alpha=alpha)


def get_projected_track_position(track_graph, track_segment_id, position):
    track_segment_id[np.isnan(track_segment_id)] = 0
    track_segment_id = track_segment_id.astype(int)

    track_segments = get_track_segments_from_graph(track_graph)
    projected_track_position = project_points_to_segment(
        track_segments, position)
    n_time = projected_track_position.shape[0]
    return projected_track_position[(
        np.arange(n_time), track_segment_id)]


def make_linearization_movie(epoch_key, animals, max_distance_from_well=5,
                             route_euclidean_distance_scaling=1,
                             min_distance_traveled=50,
                             sensor_std_dev=10, spacing=30):
    animal, day, epoch = epoch_key
    position_info = get_position_dataframe(
        epoch_key, animals, use_hmm=True,
        max_distance_from_well=max_distance_from_well,
        route_euclidean_distance_scaling=route_euclidean_distance_scaling,
        min_distance_traveled=min_distance_traveled,
        sensor_std_dev=sensor_std_dev,
        spacing=spacing)

    track_graph, center_well_id = make_track_graph(epoch_key, animals)
    position = position_info.loc[:, ['x_position', 'y_position']].values
    track_segment_id = position_info.track_segment_id.values
    projected_track_position = get_projected_track_position(
        track_graph, track_segment_id, position)
    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=33, metadata=dict(artist='Me'), bitrate=1800)

    fig, ax = plt.subplots(figsize=(7, 7))
    plt.plot(position[:, 0], position[:, 1], color='lightgrey', zorder=-10)
    plot_track(track_graph, ax)

    plt.xlim(position[:, 0].min() - 1, position[:, 0].max() + 1)
    plt.ylim(position[:, 1].min() + 1, position[:, 1].max() + 1)
    sns.despine(left=True, bottom=True, ax=ax)
    plt.title('Linearized vs. Actual Position')

    actual_line, = plt.plot(
        [], [], 'g-', label='actual position', linewidth=3, zorder=101)
    actual_head = plt.scatter([], [], s=80, zorder=101, color='g')

    predicted_line, = plt.plot(
        [], [], 'r-', label='linearized position', linewidth=3, zorder=102)
    predicted_head = plt.scatter([], [], s=80, zorder=102, color='r')

    plt.legend()

    def _update_line(time_ind):
        start_ind = max(0, time_ind - 33)
        time_slice = slice(start_ind, time_ind)

        actual_line.set_data(position[time_slice, 0], position[time_slice, 1])
        actual_head.set_offsets(position[time_ind])

        predicted_line.set_data(projected_track_position[time_slice, 0],
                                projected_track_position[time_slice, 1])
        predicted_head.set_offsets(projected_track_position[time_ind])

        return actual_line, predicted_line

    n_time = position.shape[0]
    line_ani = animation.FuncAnimation(fig, _update_line, frames=n_time,
                                       interval=50, blit=True)
    line_ani.save(
        f'{animal}_{day:02}_{epoch:02}_linearization.mp4', writer=writer)
