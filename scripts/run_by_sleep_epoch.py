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
from replay_trajectory_classification import (ClusterlessClassifier,
                                              SortedSpikesClassifier)
from scipy.ndimage import label
from src.analysis import (get_place_field_max, get_sleep_replay_info,
                          reshape_to_segments)
from src.load_data import get_sleep_and_prev_run_epochs, load_sleep_data
from src.parameters import (FIGURE_DIR, PROBABILITY_THRESHOLD,
                            PROCESSED_DATA_DIR, SAMPLING_FREQUENCY,
                            TRANSITION_TO_CATEGORY)
from src.visualization import (plot_category_counts, plot_category_duration,
                               plot_ripple_decode_1D)
from tqdm.auto import tqdm

FORMAT = '%(asctime)s %(message)s'

logging.basicConfig(level='INFO', format=FORMAT, datefmt='%d-%b-%y %H:%M:%S')
plt.switch_backend('agg')


def sorted_spikes_analysis_1D(sleep_epoch_key, prev_run_epoch_key,
                              plot_ripple_figures=False):
    data_type, dim = 'sorted_spikes', '1D'

    logging.info('Loading data...')
    data = load_sleep_data(sleep_epoch_key)

    model_name = os.path.join(
        PROCESSED_DATA_DIR,
        (f'{prev_run_epoch_key[0]}_{prev_run_epoch_key[1]:02}_'
         f'{prev_run_epoch_key[2]:02}_{data_type}_{dim}_model.pkl'))
    try:
        results = xr.open_dataset(
            os.path.join(
                PROCESSED_DATA_DIR,
                (f'{sleep_epoch_key[0]}_{sleep_epoch_key[1]:02}'
                 f'_{sleep_epoch_key[2]:02}.nc')),
            group=f'/{data_type}/{dim}/classifier/ripples/')
        logging.info('Found existing results. Loading...')
        ripple_times = data['ripple_times'].loc[:, ['start_time', 'end_time']]
        ripple_spikes = reshape_to_segments(data['spikes'], ripple_times)
        classifier = SortedSpikesClassifier.load_model(model_name)
        logging.info(classifier)
    except (FileNotFoundError, OSError):
        classifier = SortedSpikesClassifier.load_model(model_name)
        logging.info(classifier)

        # Decode
        is_test = data['position_info'].speed <= 4

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
        save_xarray(PROCESSED_DATA_DIR, sleep_epoch_key,
                    results.drop(['likelihood', 'causal_posterior']),
                    group=f'/{data_type}/{dim}/classifier/ripples/')

    logging.info('Saving replay_info...')
    replay_info = get_sleep_replay_info(
        results, ripple_spikes, data['ripple_times'], data['position_info'],
        SAMPLING_FREQUENCY, PROBABILITY_THRESHOLD, sleep_epoch_key,
        classifier)
    epoch_identifier = (f'{sleep_epoch_key[0]}_{sleep_epoch_key[1]:02d}'
                        f'_{sleep_epoch_key[2]:02d}_{data_type}_{dim}')
    replay_info_filename = os.path.join(
        PROCESSED_DATA_DIR, f'{epoch_identifier}_replay_info_sleep.csv')
    replay_info.to_csv(replay_info_filename)

    logging.info('Plotting ripple figures...')

    plot_category_counts(replay_info)
    plt.suptitle(f'Category counts - {epoch_identifier}')
    fig_name = (f'{epoch_identifier}_category_counts.png')
    fig_name = os.path.join(
        FIGURE_DIR, 'sleep_category_counts_duration', fig_name)
    plt.savefig(fig_name, bbox_inches='tight')
    plt.close(plt.gcf())

    plot_category_duration(replay_info)
    plt.suptitle(f'Category duration - {epoch_identifier}')
    fig_name = (f'{epoch_identifier}_category_duration.png')
    fig_name = os.path.join(
        FIGURE_DIR, 'sleep_category_counts_duration', fig_name)
    plt.savefig(fig_name, bbox_inches='tight')
    plt.close(plt.gcf())

    if plot_ripple_figures:
        place_field_max = get_place_field_max(classifier)
        linear_position_order = place_field_max.argsort(axis=0).squeeze()

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
                ripple_position = np.full_like(posterior.time.values, np.nan)
                plot_ripple_decode_1D(
                    posterior, ripple_position,
                    ripple_spikes.loc[ripple_number], linear_position_order,
                    data['position_info'], classifier)
                plt.suptitle(
                    f'ripple number = {sleep_epoch_key[0]}_'
                    f'{sleep_epoch_key[1]:02d}_{sleep_epoch_key[2]:02d}'
                    f'_{ripple_number:04d}')
                fig_name = (
                    f'{sleep_epoch_key[0]}_{sleep_epoch_key[1]:02d}_'
                    f'{sleep_epoch_key[2]:02d}_{ripple_number:04d}_'
                    f'{data_type}_{dim}_acasual_classification.png')
                fig_name = os.path.join(
                    FIGURE_DIR, 'sleep_ripple_classifications', fig_name)
                plt.savefig(fig_name, bbox_inches='tight')
                plt.close(plt.gcf())
            except (ValueError, IndexError):
                logging.warn(f'No figure for ripple number {ripple_number}...')
                continue

    logging.info('Done...')


def clusterless_analysis_1D(sleep_epoch_key, prev_run_epoch_key,
                            plot_ripple_figures=False):
    data_type, dim = 'clusterless', '1D'

    logging.info('Loading data...')
    data = load_sleep_data(sleep_epoch_key)

    model_name = os.path.join(
        PROCESSED_DATA_DIR,
        (f'{prev_run_epoch_key[0]}_{prev_run_epoch_key[1]:02}_'
         f'{prev_run_epoch_key[2]:02}_{data_type}_{dim}_model.pkl'))
    try:
        results = xr.open_dataset(
            os.path.join(
                PROCESSED_DATA_DIR,
                (f'{sleep_epoch_key[0]}_{sleep_epoch_key[1]:02}'
                 f'_{sleep_epoch_key[2]:02}.nc')),
            group=f'/{data_type}/{dim}/classifier/ripples/')
        logging.info('Found existing results. Loading...')
        ripple_times = data['ripple_times'].loc[:, ['start_time', 'end_time']]
        spikes = (((data['multiunit'].sum('features') > 0) * 1.0)
                  .to_dataframe(name='spikes').unstack())
        spikes.columns = data['tetrode_info'].tetrode_id
        ripple_spikes = reshape_to_segments(spikes, ripple_times)
        classifier = ClusterlessClassifier.load_model(model_name)
        logging.info(classifier)
    except (FileNotFoundError, OSError):
        classifier = ClusterlessClassifier.load_model(model_name)
        logging.info(classifier)

        # Decode
        is_test = data['position_info'].speed <= 4

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
        spikes = (((data['multiunit'].sum('features') > 0) * 1.0)
                  .to_dataframe(name='spikes').unstack())
        spikes.columns = data['tetrode_info'].tetrode_id
        ripple_spikes = reshape_to_segments(spikes, ripple_times)

        logging.info('Saving results...')
        save_xarray(PROCESSED_DATA_DIR, sleep_epoch_key,
                    results.drop(['likelihood', 'causal_posterior']),
                    group=f'/{data_type}/{dim}/classifier/ripples/')

    logging.info('Saving replay_info...')
    replay_info = get_sleep_replay_info(
        results, ripple_spikes, data['ripple_times'], data['position_info'],
        SAMPLING_FREQUENCY, PROBABILITY_THRESHOLD, sleep_epoch_key, classifier)
    epoch_identifier = (f'{sleep_epoch_key[0]}_{sleep_epoch_key[1]:02d}'
                        f'_{sleep_epoch_key[2]:02d}_{data_type}_{dim}')
    replay_info_filename = os.path.join(
        PROCESSED_DATA_DIR, f'{epoch_identifier}_replay_info_sleep.csv')
    replay_info.to_csv(replay_info_filename)

    logging.info('Plotting ripple figures...')

    plot_category_counts(replay_info)
    plt.suptitle(f'Category counts - {epoch_identifier}')
    fig_name = (f'{epoch_identifier}_category_counts.png')
    fig_name = os.path.join(
        FIGURE_DIR, 'sleep_category_counts_duration', fig_name)
    plt.savefig(fig_name, bbox_inches='tight')
    plt.close(plt.gcf())

    plot_category_duration(replay_info)
    plt.suptitle(f'Category duration - {epoch_identifier}')
    fig_name = (f'{epoch_identifier}_category_duration.png')
    fig_name = os.path.join(
        FIGURE_DIR, 'sleep_category_counts_duration', fig_name)
    plt.savefig(fig_name, bbox_inches='tight')
    plt.close(plt.gcf())

    if plot_ripple_figures:
        place_field_max = get_place_field_max(classifier)
        linear_position_order = place_field_max.argsort(axis=0).squeeze()
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
                ripple_position = np.full_like(posterior.time.values, np.nan)
                plot_ripple_decode_1D(
                    posterior, ripple_position,
                    ripple_spikes.loc[ripple_number], linear_position_order,
                    data['position_info'], classifier, spike_label='Tetrodes')
                plt.suptitle(
                    f'ripple number = {sleep_epoch_key[0]}_'
                    f'{sleep_epoch_key[1]:02d}_{sleep_epoch_key[2]:02d}'
                    f'_{ripple_number:04d}')
                fig_name = (
                    f'{sleep_epoch_key[0]}_{sleep_epoch_key[1]:02d}_'
                    f'{sleep_epoch_key[2]:02d}_{ripple_number:04d}_'
                    f'{data_type}_{dim}_acasual_classification.png')
                fig_name = os.path.join(
                    FIGURE_DIR, 'sleep_ripple_classifications', fig_name)
                plt.savefig(fig_name, bbox_inches='tight')
                plt.close(plt.gcf())
            except (ValueError, IndexError):
                logging.warn(f'No figure for ripple number {ripple_number}...')
                continue

    logging.info('Done...')


run_analysis = {
    ('sorted_spikes', '1D'): sorted_spikes_analysis_1D,
    ('clusterless', '1D'): clusterless_analysis_1D,
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

    sleep_epoch_key = (args.Animal, args.Day, args.Epoch)
    sleep_epoch_keys, prev_run_epoch_keys = get_sleep_and_prev_run_epochs()
    prev_run_epoch_key = prev_run_epoch_keys[
        sleep_epoch_keys.index(sleep_epoch_key)]
    logging.info(
        'Processing epoch: Animal {0}, Day {1}, Epoch #{2}...'.format(
            *sleep_epoch_key))
    logging.info(f'Data type: {args.data_type}, Dim: {args.dim}')
    git_hash = run(['git', 'rev-parse', 'HEAD'],
                   stdout=PIPE, universal_newlines=True).stdout
    logging.info('Git Hash: {git_hash}'.format(git_hash=git_hash.rstrip()))

    # Analysis Code
    run_analysis[(args.data_type, args.dim)](
        sleep_epoch_key, prev_run_epoch_key,
        plot_ripple_figures=args.plot_ripple_figures)


if __name__ == '__main__':
    sys.exit(main())
