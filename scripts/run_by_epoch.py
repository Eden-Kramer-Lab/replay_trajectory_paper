import logging
import os
import sys
from argparse import ArgumentParser
from signal import SIGUSR1, SIGUSR2, signal
from subprocess import PIPE, run

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from loren_frank_data_processing import save_xarray
from loren_frank_data_processing.position import (EDGE_ORDER, EDGE_SPACING,
                                                  make_track_graph)
from replay_trajectory_classification import (ClusterlessClassifier,
                                              SortedSpikesClassifier)
from scipy.ndimage import label
from src.analysis import (get_linear_position_order, get_place_field_max,
                          get_replay_info, reshape_to_segments)
from src.load_data import load_data
from src.parameters import (ANIMALS, FIGURE_DIR, PROBABILITY_THRESHOLD,
                            PROCESSED_DATA_DIR, SAMPLING_FREQUENCY,
                            TRANSITION_TO_CATEGORY,
                            continuous_transition_types, discrete_diag,
                            knot_spacing, model, model_kwargs, movement_var,
                            place_bin_size, replay_speed, spike_model_penalty)
from src.visualization import (plot_category_counts, plot_category_duration,
                               plot_neuron_place_field_2D_1D_position,
                               plot_ripple_decode_1D, plot_ripple_decode_2D)
from tqdm.auto import tqdm

FORMAT = '%(asctime)s %(message)s'

logging.basicConfig(level='INFO', format=FORMAT, datefmt='%d-%b-%y %H:%M:%S')
plt.switch_backend('agg')


def sorted_spikes_analysis_1D(epoch_key, plot_ripple_figures=False,
                              exclude_interneuron_spikes=False):
    animal, day, epoch = epoch_key
    data_type, dim = 'sorted_spikes', '1D'

    logging.info('Loading data...')
    data = load_data(epoch_key,
                     exclude_interneuron_spikes=exclude_interneuron_spikes)
    is_training = data['position_info'].speed > 4
    position = data['position_info'].loc[:, 'linear_position']
    track_graph, center_well_id = make_track_graph(epoch_key, ANIMALS)

    model_name = os.path.join(
        PROCESSED_DATA_DIR,
        f'{animal}_{day:02}_{epoch:02}_{data_type}_{dim}_model.pkl')
    try:
        results = xr.open_dataset(
            os.path.join(
                PROCESSED_DATA_DIR, f'{animal}_{day:02}_{epoch:02}.nc'),
            group=f'/{data_type}/{dim}/classifier/ripples/')
        logging.info('Found existing results. Loading...')
        ripple_times = data['ripple_times'].loc[:, ['start_time', 'end_time']]
        ripple_spikes = reshape_to_segments(data['spikes'], ripple_times)
        classifier = SortedSpikesClassifier.load_model(model_name)
        logging.info(classifier)
    except (FileNotFoundError, OSError):
        logging.info('Fitting classifier...')
        classifier = SortedSpikesClassifier(
            place_bin_size=place_bin_size, movement_var=movement_var,
            replay_speed=replay_speed,
            discrete_transition_diag=discrete_diag,
            spike_model_penalty=spike_model_penalty, knot_spacing=knot_spacing,
            continuous_transition_types=continuous_transition_types).fit(
                position, data['spikes'], is_training=is_training,
                track_graph=track_graph, center_well_id=center_well_id,
                edge_order=EDGE_ORDER, edge_spacing=EDGE_SPACING)
        classifier.save_model(model_name)
        logging.info(classifier)

        # Plot Place Fields
        g = (classifier.place_fields_ * data['sampling_frequency']).plot(
            x='position', col='neuron', col_wrap=4)
        arm_grouper = (data['position_info']
                       .groupby('arm_name')
                       .linear_position)
        max_df = arm_grouper.max()
        min_df = arm_grouper.min()
        plt.xlim((0, data['position_info'].linear_position.max()))
        max_rate = (classifier.place_fields_.values.max() *
                    data['sampling_frequency'])
        for ax in g.axes.flat:
            for arm_name, min_position in min_df.iteritems():
                ax.axvline(min_position, color='lightgrey', zorder=0,
                           linestyle='--')
                ax.text(min_position + 0.2, max_rate, arm_name,
                        color='lightgrey', horizontalalignment='left',
                        verticalalignment='top', fontsize=8)
            for arm_name, max_position in max_df.iteritems():
                ax.axvline(max_position, color='lightgrey', zorder=0,
                           linestyle='--')
        plt.suptitle(epoch_key, y=1.04, fontsize=16)
        fig_name = (
            f'{animal}_{day:02d}_{epoch:02d}_{data_type}_place_fields_1D.png')
        fig_name = os.path.join(FIGURE_DIR, 'neuron_place_fields', fig_name)
        plt.savefig(fig_name, bbox_inches='tight')
        plt.close(g.fig)

        # Decode
        is_test = ~is_training

        test_groups = pd.DataFrame(
            {'test_groups': label(is_test.values)[0]}, index=is_test.index)
        immobility_results = []
        for _, df in tqdm(test_groups.loc[is_test].groupby('test_groups'),
                          desc='immobility'):
            start_time, end_time = df.iloc[0].name, df.iloc[-1].name
            test_spikes = data['spikes'].loc[start_time:end_time]
            immobility_results.append(
                classifier.predict(test_spikes, time=test_spikes.index))
        immobility_results = xr.concat(immobility_results, dim='time')

        results = [(immobility_results
                    .sel(time=slice(df.start_time, df.end_time))
                    .assign_coords(time=lambda ds: ds.time - ds.time[0]))
                   for _, df in data['ripple_times'].iterrows()]

        results = (xr.concat(results, dim=data['ripple_times'].index)
                   .assign_coords(state=lambda ds: ds.state.to_index()
                                  .map(TRANSITION_TO_CATEGORY)))

        logging.info('Saving results...')
        ripple_times = data['ripple_times'].loc[:, ['start_time', 'end_time']]
        ripple_spikes = reshape_to_segments(data['spikes'], ripple_times)
        save_xarray(PROCESSED_DATA_DIR, epoch_key,
                    results.drop(['likelihood', 'causal_posterior']),
                    group=f'/{data_type}/{dim}/classifier/ripples/')

    logging.info('Saving replay_info...')
    replay_info = get_replay_info(
        results, data['spikes'], data['ripple_times'], data['position_info'],
        track_graph, SAMPLING_FREQUENCY, PROBABILITY_THRESHOLD, epoch_key,
        classifier, data["ripple_consensus_trace_zscore"])
    prob = int(PROBABILITY_THRESHOLD * 100)
    epoch_identifier = f'{animal}_{day:02d}_{epoch:02d}_{data_type}_{dim}'
    replay_info_filename = os.path.join(
        PROCESSED_DATA_DIR, f'{epoch_identifier}_replay_info_{prob:02d}.csv')
    replay_info.to_csv(replay_info_filename)

    logging.info('Plotting ripple figures...')

    if plot_ripple_figures:
        place_field_max = get_place_field_max(classifier)
        linear_position_order = place_field_max.argsort(axis=0).squeeze()
        ripple_position = reshape_to_segments(position, ripple_times)

        for ripple_number in tqdm(ripple_times.index, desc='ripple figures'):
            try:
                posterior = (
                    results
                    .acausal_posterior
                    .sel(ripple_number=ripple_number)
                    .dropna('time', how='all')
                    .assign_coords(
                        time=lambda ds: 1000 * ds.time /
                        np.timedelta64(1, 's')))
                plot_ripple_decode_1D(
                    posterior, ripple_position.loc[ripple_number],
                    ripple_spikes.loc[ripple_number], linear_position_order,
                    data['position_info'], classifier)
                plt.suptitle(
                    f'ripple number = {animal}_{day:02d}_{epoch:02d}_'
                    f'{ripple_number:04d}')
                fig_name = (
                    f'{animal}_{day:02d}_{epoch:02d}_{ripple_number:04d}_'
                    f'{data_type}_{dim}_acasual_classification.png')
                fig_name = os.path.join(
                    FIGURE_DIR, 'ripple_classifications', fig_name)
                plt.savefig(fig_name, bbox_inches='tight')
                plt.close(plt.gcf())
            except (ValueError, IndexError):
                logging.warn(f'No figure for ripple number {ripple_number}...')
                continue

    logging.info('Done...')


def sorted_spikes_analysis_2D(epoch_key, plot_ripple_figures=False,
                              exclude_interneuron_spikes=False):
    animal, day, epoch = epoch_key
    data_type, dim = 'sorted_spikes', '2D'

    logging.info('Loading data...')
    data = load_data(epoch_key,
                     exclude_interneuron_spikes=exclude_interneuron_spikes)

    is_training = data['position_info'].speed > 4
    position = data['position_info'].loc[:, ['x_position', 'y_position']]

    model_name = os.path.join(
        PROCESSED_DATA_DIR,
        f'{animal}_{day:02}_{epoch:02}_{data_type}_{dim}_model.pkl')
    try:
        results = xr.open_dataset(
            os.path.join(
                PROCESSED_DATA_DIR, f'{animal}_{day:02}_{epoch:02}.nc'),
            group=f'/{data_type}/{dim}/classifier/ripples/')
        logging.info('Found existing results. Loading...')
        ripple_times = data['ripple_times'].loc[:, ['start_time', 'end_time']]
        ripple_spikes = reshape_to_segments(data['spikes'], ripple_times)
        classifier = SortedSpikesClassifier.load_model(model_name)
        logging.info(classifier)
    except (FileNotFoundError, OSError):
        logging.info('Fitting classifier...')
        classifier = SortedSpikesClassifier(
            place_bin_size=place_bin_size, movement_var=movement_var,
            replay_speed=replay_speed,
            discrete_transition_diag=discrete_diag,
            spike_model_penalty=spike_model_penalty, knot_spacing=knot_spacing,
            continuous_transition_types=continuous_transition_types).fit(
            position, data['spikes'], is_training=is_training)
        classifier.save_model(model_name)
        logging.info(classifier)

        # Plot Place Fields
        g = classifier.plot_place_fields(
            data['spikes'], position, SAMPLING_FREQUENCY)
        plt.suptitle(epoch_key, y=1.04, fontsize=16)

        fig_name = (
            f'{animal}_{day:02d}_{epoch:02d}_{data_type}_place_fields_{dim}'
            '.png')
        fig_name = os.path.join(FIGURE_DIR, 'neuron_place_fields', fig_name)
        plt.savefig(fig_name, bbox_inches='tight')
        plt.close(g.fig)

        # Decode
        ripple_times = data['ripple_times'].loc[:, ['start_time', 'end_time']]
        ripple_spikes = reshape_to_segments(data['spikes'], ripple_times)
        results = []
        for ripple_number in tqdm(data['ripple_times'].index, desc='ripple'):
            ripple_time = (ripple_spikes.loc[ripple_number].index -
                           ripple_spikes.loc[ripple_number].index[0])
            results.append(
                classifier.predict(ripple_spikes.loc[ripple_number],
                                   time=ripple_time))
        results = (xr.concat(results, dim=data['ripple_times'].index)
                   .assign_coords(state=lambda ds: ds.state.to_index()
                                  .map(TRANSITION_TO_CATEGORY)))

        logging.info('Saving results...')
        save_xarray(PROCESSED_DATA_DIR, epoch_key,
                    results.drop(['likelihood', 'causal_posterior']),
                    group=f'/{data_type}/{dim}/classifier/ripples/')

    logging.info('Saving replay_info...')
    track_graph, _ = make_track_graph(epoch_key, ANIMALS)
    replay_info = get_replay_info(
        results, ripple_spikes, data['ripple_times'], data['position_info'],
        track_graph, SAMPLING_FREQUENCY, PROBABILITY_THRESHOLD, epoch_key,
        classifier)
    epoch_identifier = f'{animal}_{day:02d}_{epoch:02d}_{data_type}_{dim}'
    prob = int(PROBABILITY_THRESHOLD * 100)
    replay_info_filename = os.path.join(
        PROCESSED_DATA_DIR, f'{epoch_identifier}_replay_info_{prob:02d}.csv')
    replay_info.to_csv(replay_info_filename)

    logging.info('Plotting ripple figures...')

    plot_category_counts(replay_info)
    plt.suptitle(f'Category counts - {animal}_{day:02d}_{epoch:02d}')
    fig_name = (f'{epoch_identifier}_category_counts.png')
    fig_name = os.path.join(FIGURE_DIR, fig_name)
    plt.savefig(fig_name, bbox_inches='tight')
    plt.close(plt.gcf())

    plot_category_duration(replay_info)
    plt.suptitle(f'Category duration - {animal}_{day:02d}_{epoch:02d}')
    fig_name = (f'{epoch_identifier}_category_duration.png')
    fig_name = os.path.join(FIGURE_DIR, fig_name)
    plt.savefig(fig_name, bbox_inches='tight')
    plt.close(plt.gcf())

    if plot_ripple_figures:
        place_field_max = get_place_field_max(classifier)
        (linear_position_order,
         linear_place_field_max) = get_linear_position_order(
            data['position_info'], place_field_max)
        plot_neuron_place_field_2D_1D_position(
            data['position_info'], place_field_max, linear_place_field_max,
            linear_position_order)
        fig_name = (f'{epoch_identifier}_place_field_max.png')
        fig_name = os.path.join(FIGURE_DIR, 'neuron_place_fields', fig_name)
        plt.savefig(fig_name, bbox_inches='tight')
        plt.close(plt.gcf())

        ripple_position = reshape_to_segments(position, ripple_times)

        for ripple_number in tqdm(ripple_times.index, desc='ripple figures'):
            try:
                posterior = (
                    results
                    .acausal_posterior
                    .sel(ripple_number=ripple_number)
                    .dropna('time', how='all')
                    .assign_coords(
                        time=lambda ds: 1000 * ds.time /
                        np.timedelta64(1, 's'),))
                plot_ripple_decode_2D(
                    posterior, ripple_position.loc[ripple_number],
                    ripple_spikes.loc[ripple_number], linear_position_order,
                    data['position_info'], spike_label='Cells')
                plt.suptitle(
                    f'ripple number = {animal}_{day:02d}_{epoch:02d}_'
                    f'{ripple_number:04d}')
                fig_name = (
                    f'{animal}_{day:02d}_{epoch:02d}_{ripple_number:04d}_'
                    f'{data_type}_{dim}_acasual_classification.png')
                fig_name = os.path.join(
                    FIGURE_DIR, 'ripple_classifications', fig_name)
                plt.savefig(fig_name, bbox_inches='tight')
                plt.close(plt.gcf())
            except (ValueError, IndexError):
                logging.warn(f'No figure for ripple number {ripple_number}...')
                pass
    logging.info('Done...')


def clusterless_analysis_1D(epoch_key, plot_ripple_figures=False,
                            exclude_interneuron_spikes=False):
    animal, day, epoch = epoch_key
    data_type, dim = 'clusterless', '1D'

    logging.info('Loading data...')
    data = load_data(epoch_key,
                     exclude_interneuron_spikes=exclude_interneuron_spikes)

    is_training = data['position_info'].speed > 4
    position = data['position_info'].loc[:, 'linear_position']
    track_graph, center_well_id = make_track_graph(epoch_key, ANIMALS)

    if not exclude_interneuron_spikes:
        model_name = os.path.join(
            PROCESSED_DATA_DIR,
            f'{animal}_{day:02}_{epoch:02}_{data_type}_{dim}_model.pkl')
        group = f'/{data_type}/{dim}/classifier/ripples/'
        epoch_identifier = f'{animal}_{day:02d}_{epoch:02d}_{data_type}_{dim}'
    else:
        model_name = os.path.join(
            PROCESSED_DATA_DIR,
            f'{animal}_{day:02}_{epoch:02}_{data_type}_{dim}'
            '_no_interneuron_model.pkl')
        group = f'/{data_type}/{dim}/no_interneuron/classifier/ripples/'
        epoch_identifier = (f'{animal}_{day:02d}_{epoch:02d}_{data_type}_{dim}'
                            '_no_interneuron')

    try:
        results = xr.open_dataset(
            os.path.join(
                PROCESSED_DATA_DIR, f'{animal}_{day:02}_{epoch:02}.nc'),
            group=group)
        logging.info('Found existing results. Loading...')
        ripple_times = data['ripple_times'].loc[:, ['start_time', 'end_time']]
        spikes = (((data['multiunit'].sum('features') > 0) * 1.0)
                  .to_dataframe(name='spikes').unstack())
        spikes.columns = data['tetrode_info'].tetrode_id
        ripple_spikes = reshape_to_segments(spikes, ripple_times)
        classifier = ClusterlessClassifier.load_model(model_name)
        logging.info(classifier)
    except (FileNotFoundError, OSError):
        logging.info('Fitting classifier...')
        classifier = ClusterlessClassifier(
            place_bin_size=place_bin_size, movement_var=movement_var,
            replay_speed=replay_speed,
            discrete_transition_diag=discrete_diag,
            continuous_transition_types=continuous_transition_types,
            model=model, model_kwargs=model_kwargs).fit(
                position, data['multiunit'], is_training=is_training,
                track_graph=track_graph, center_well_id=center_well_id,
                edge_order=EDGE_ORDER, edge_spacing=EDGE_SPACING)
        classifier.save_model(model_name)
        logging.info(classifier)

        # Decode
        is_test = ~is_training

        test_groups = pd.DataFrame(
            {'test_groups': label(is_test.values)[0]}, index=is_test.index)
        immobility_results = []
        for _, df in tqdm(test_groups.loc[is_test].groupby('test_groups'),
                          desc='immobility'):
            start_time, end_time = df.iloc[0].name, df.iloc[-1].name
            test_multiunit = data['multiunit'].sel(
                time=slice(start_time, end_time))
            immobility_results.append(
                classifier.predict(test_multiunit, time=test_multiunit.time))

        immobility_results = xr.concat(immobility_results, dim='time')

        results = [(immobility_results
                    .sel(time=slice(df.start_time, df.end_time))
                    .assign_coords(time=lambda ds: ds.time - ds.time[0]))
                   for _, df in data['ripple_times'].iterrows()]

        results = (xr.concat(results, dim=data['ripple_times'].index)
                   .assign_coords(state=lambda ds: ds.state.to_index()
                                  .map(TRANSITION_TO_CATEGORY)))

        ripple_times = data['ripple_times'].loc[:, ['start_time', 'end_time']]
        spikes = ((((~np.isnan(data['multiunit'])).sum('features') > 0) * 1.0)
                  .to_dataframe(name='spikes').unstack())
        spikes.columns = data['tetrode_info'].tetrode_id
        ripple_spikes = reshape_to_segments(spikes, ripple_times)

        logging.info('Saving results...')
        save_xarray(PROCESSED_DATA_DIR, epoch_key,
                    results.drop(['likelihood', 'causal_posterior']),
                    group=group)

    logging.info('Saving replay_info...')
    replay_info = get_replay_info(
        results, spikes, data['ripple_times'], data['position_info'],
        track_graph, SAMPLING_FREQUENCY, PROBABILITY_THRESHOLD, epoch_key,
        classifier, data["ripple_consensus_trace_zscore"])
    prob = int(PROBABILITY_THRESHOLD * 100)
    replay_info_filename = os.path.join(
        PROCESSED_DATA_DIR, f'{epoch_identifier}_replay_info_{prob:02d}.csv')
    replay_info.to_csv(replay_info_filename)

    logging.info('Plotting ripple figures...')

    if plot_ripple_figures:
        place_field_max = get_place_field_max(classifier)
        linear_position_order = place_field_max.argsort(axis=0).squeeze()
        ripple_position = reshape_to_segments(position, ripple_times)

        for ripple_number in tqdm(ripple_times.index, desc='ripple figures'):
            try:
                posterior = (
                    results
                    .acausal_posterior
                    .sel(ripple_number=ripple_number)
                    .dropna('time', how='all')
                    .assign_coords(
                        time=lambda ds: 1000 * ds.time /
                        np.timedelta64(1, 's')))
                plot_ripple_decode_1D(
                    posterior, ripple_position.loc[ripple_number],
                    ripple_spikes.loc[ripple_number], linear_position_order,
                    data['position_info'], classifier, spike_label='Tetrodes')
                plt.suptitle(
                    f'ripple number = {animal}_{day:02d}_{epoch:02d}_'
                    f'{ripple_number:04d}')
                fig_name = (
                    f'{animal}_{day:02d}_{epoch:02d}_{ripple_number:04d}_'
                    f'{data_type}_{dim}_acasual_classification.png')
                fig_name = os.path.join(
                    FIGURE_DIR, 'ripple_classifications', fig_name)
                plt.savefig(fig_name, bbox_inches='tight')
                plt.close(plt.gcf())
            except (ValueError, IndexError):
                logging.warn(f'No figure for ripple number {ripple_number}...')
                continue

    logging.info('Done...')


def clusterless_analysis_2D(epoch_key, plot_ripple_figures=False,
                            exclude_interneuron_spikes=False):
    animal, day, epoch = epoch_key
    data_type, dim = 'clusterless', '2D'

    logging.info('Loading data...')
    data = load_data(epoch_key,
                     exclude_interneuron_spikes=exclude_interneuron_spikes)

    position = data['position_info'].loc[:, ['x_position', 'y_position']]
    is_training = data['position_info'].speed > 4

    model_name = os.path.join(
        PROCESSED_DATA_DIR,
        f'{animal}_{day:02}_{epoch:02}_{data_type}_{dim}_model.pkl')
    try:
        results = xr.open_dataset(
            os.path.join(
                PROCESSED_DATA_DIR, f'{animal}_{day:02}_{epoch:02}.nc'),
            group=f'/{data_type}/{dim}/classifier/ripples/')
        logging.info('Found existing results. Loading...')
        ripple_times = data['ripple_times'].loc[:, ['start_time', 'end_time']]
        spikes = ((((~np.isnan(data['multiunit'])).sum('features') > 0) * 1.0)
                  .to_dataframe(name='spikes').unstack())
        spikes.columns = data['tetrode_info'].tetrode_id
        ripple_spikes = reshape_to_segments(spikes, ripple_times)
        classifier = ClusterlessClassifier.load_model(model_name)
        logging.info(classifier)
    except (FileNotFoundError, OSError):
        logging.info('Fitting classifier...')
        classifier = ClusterlessClassifier(
            place_bin_size=place_bin_size, movement_var=movement_var,
            replay_speed=replay_speed,
            discrete_transition_diag=discrete_diag,
            continuous_transition_types=continuous_transition_types,
            model=model, model_kwargs=model_kwargs).fit(
            position, data['multiunit'], is_training=is_training)
        classifier.save_model(model_name)
        logging.info(classifier)
        # Decode
        ripple_times = data['ripple_times'].loc[:, ['start_time', 'end_time']]
        spikes = (((data['multiunit'].sum('features') > 0) * 1.0)
                  .to_dataframe(name='spikes').unstack())
        spikes.columns = data['tetrode_info'].tetrode_id
        ripple_spikes = reshape_to_segments(spikes, ripple_times)

        results = []
        for ripple_number in tqdm(data['ripple_times'].index, desc='ripple'):
            time_slice = slice(*data['ripple_times'].loc[
                ripple_number, ['start_time', 'end_time']])
            m = data['multiunit'].sel(time=time_slice)
            results.append(classifier.predict(m, m.time - m.time[0]))
        results = xr.concat(results, dim=data['ripple_times'].index)
        results = results.assign_coords(
            state=lambda ds: ds.state.to_index()
            .map(TRANSITION_TO_CATEGORY))

        logging.info('Saving results...')
        save_xarray(PROCESSED_DATA_DIR, epoch_key,
                    results.drop(['likelihood', 'causal_posterior']),
                    group=f'/{data_type}/{dim}/classifier/ripples/')

    logging.info('Saving replay_info...')
    track_graph, _ = make_track_graph(epoch_key, ANIMALS)
    replay_info = get_replay_info(
        results, ripple_spikes, data['ripple_times'], data['position_info'],
        track_graph, SAMPLING_FREQUENCY, PROBABILITY_THRESHOLD, epoch_key,
        classifier)
    epoch_identifier = f'{animal}_{day:02d}_{epoch:02d}_{data_type}_{dim}'
    prob = int(PROBABILITY_THRESHOLD * 100)
    replay_info_filename = os.path.join(
        PROCESSED_DATA_DIR, f'{epoch_identifier}_replay_info_{prob:02d}.csv')
    replay_info.to_csv(replay_info_filename)

    logging.info('Plotting ripple figures...')

    plot_category_counts(replay_info)
    plt.suptitle(f'Category counts - {animal}_{day:02d}_{epoch:02d}')
    fig_name = (f'{epoch_identifier}_category_counts.png')
    fig_name = os.path.join(FIGURE_DIR, fig_name)
    plt.savefig(fig_name, bbox_inches='tight')
    plt.close(plt.gcf())

    plot_category_duration(replay_info)
    plt.suptitle(f'Category duration - {animal}_{day:02d}_{epoch:02d}')
    fig_name = (f'{epoch_identifier}_category_duration.png')
    fig_name = os.path.join(FIGURE_DIR, fig_name)
    plt.savefig(fig_name, bbox_inches='tight')
    plt.close(plt.gcf())

    if plot_ripple_figures:
        place_field_max = get_place_field_max(classifier)
        (linear_position_order,
         linear_place_field_max) = get_linear_position_order(
            data['position_info'], place_field_max)
        plot_neuron_place_field_2D_1D_position(
            data['position_info'], place_field_max, linear_place_field_max,
            linear_position_order)
        fig_name = (f'{epoch_identifier}_place_field_max.png')
        fig_name = os.path.join(FIGURE_DIR, 'neuron_place_fields', fig_name)
        plt.savefig(fig_name, bbox_inches='tight')
        plt.close(plt.gcf())

        ripple_position = reshape_to_segments(position, ripple_times)

        for ripple_number in tqdm(ripple_times.index, desc='ripple figures'):
            try:
                posterior = (
                    results
                    .acausal_posterior
                    .sel(ripple_number=ripple_number)
                    .dropna('time', how='all')
                    .assign_coords(
                        time=lambda ds: 1000 * ds.time /
                        np.timedelta64(1, 's')))
                plot_ripple_decode_2D(
                    posterior, ripple_position.loc[ripple_number],
                    ripple_spikes.loc[ripple_number], position,
                    linear_position_order, spike_label='Tetrodes')
                plt.suptitle(
                    f'ripple number = {animal}_{day:02d}_{epoch:02d}_'
                    f'{ripple_number:04d}')
                fig_name = (
                    f'{animal}_{day:02d}_{epoch:02d}_{ripple_number:04d}_'
                    f'{data_type}_{dim}_acasual_classification.png')
                fig_name = os.path.join(
                    FIGURE_DIR, 'ripple_classifications', fig_name)
                plt.savefig(fig_name, bbox_inches='tight')
                plt.close(plt.gcf())
            except (ValueError, IndexError):
                logging.warn(f'No figure for ripple number {ripple_number}...')
                continue

    logging.info('Done...')


run_analysis = {
    ('sorted_spikes', '1D'): sorted_spikes_analysis_1D,
    ('sorted_spikes', '2D'): sorted_spikes_analysis_2D,
    ('clusterless', '1D'): clusterless_analysis_1D,
    ('clusterless', '2D'): clusterless_analysis_2D,
}


def get_command_line_arguments():
    parser = ArgumentParser()
    parser.add_argument('Animal', type=str, help='Short name of animal')
    parser.add_argument('Day', type=int, help='Day of recording session')
    parser.add_argument('Epoch', type=int,
                        help='Epoch number of recording session')
    parser.add_argument('--data_type', type=str, default='sorted_spikes')
    parser.add_argument('--dim', type=str, default='1D')
    parser.add_argument('--n_workers', type=int, default=16)
    parser.add_argument('--threads_per_worker', type=int, default=1)
    parser.add_argument('--plot_ripple_figures', action='store_true')
    parser.add_argument('--exclude_interneuron_spikes', action='store_true')
    parser.add_argument(
        '-d', '--debug',
        help='More verbose output for debugging',
        action='store_const',
        dest='log_level',
        const=logging.DEBUG,
        default=logging.INFO,
    )
    return parser.parse_args()


def main():
    args = get_command_line_arguments()
    FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(format=FORMAT, level=args.log_level)

    def _signal_handler(signal_code, frame):
        logging.error('***Process killed with signal {signal}***'.format(
            signal=signal_code))
        exit()

    for code in [SIGUSR1, SIGUSR2]:
        signal(code, _signal_handler)

    epoch_key = (args.Animal, args.Day, args.Epoch)
    logging.info(
        'Processing epoch: Animal {0}, Day {1}, Epoch #{2}...'.format(
            *epoch_key))
    logging.info(f'Data type: {args.data_type}, Dim: {args.dim}')
    git_hash = run(['git', 'rev-parse', 'HEAD'],
                   stdout=PIPE, universal_newlines=True).stdout
    logging.info('Git Hash: {git_hash}'.format(git_hash=git_hash.rstrip()))

    # Analysis Code
    run_analysis[(args.data_type, args.dim)](
        epoch_key,
        plot_ripple_figures=args.plot_ripple_figures,
        exclude_interneuron_spikes=args.exclude_interneuron_spikes)


if __name__ == '__main__':
    sys.exit(main())
