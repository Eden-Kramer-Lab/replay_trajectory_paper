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
from replay_trajectory_classification import ClusterlessClassifier
from sklearn.model_selection import KFold
from src.figure_utilities import PAGE_HEIGHT, TWO_COLUMN
from src.load_data import load_data
from src.parameters import (_MARKS, ANIMALS, FIGURE_DIR, PROBABILITY_THRESHOLD,
                            PROCESSED_DATA_DIR, TRANSITION_TO_CATEGORY,
                            continuous_transition_types, discrete_diag, model,
                            model_kwargs, movement_var, place_bin_size,
                            replay_speed)
from src.visualization import plot_classifier_time_slice
from trajectory_analysis_tools import (get_ahead_behind_distance,
                                       get_trajectory_data)

FORMAT = '%(asctime)s %(message)s'

logging.basicConfig(level='INFO', format=FORMAT, datefmt='%d-%b-%y %H:%M:%S')


def clusterless_analysis_1D(epoch_key, plot_figures=False):
    animal, day, epoch = epoch_key
    data_type, dim = 'clusterless', '1D'

    logging.info('Loading data...')
    data = load_data(epoch_key)
    data['multiunit'] = data['multiunit'].sel(features=_MARKS)

    track_graph, center_well_id = make_track_graph(epoch_key, ANIMALS)

    cv = KFold()
    results = []
    error = []
    index = pd.Index([f"{animal}_{day:02d}_{epoch:02d}"])
    error_info = pd.DataFrame([], index=index)

    for fold_ind, (train, test) in enumerate(
            cv.split(data["position_info"].index)):
        logging.info(f'Fitting Fold #{fold_ind + 1}...')
        classifier = ClusterlessClassifier(
            place_bin_size=place_bin_size,
            movement_var=movement_var,
            replay_speed=replay_speed,
            discrete_transition_diag=discrete_diag,
            continuous_transition_types=continuous_transition_types,
            model=model,
            model_kwargs=model_kwargs)
        classifier.fit(
            position=data["position_info"].iloc[train].linear_position,
            multiunits=data["multiunit"].isel(time=train),
            track_graph=track_graph,
            center_well_id=center_well_id,
            edge_order=EDGE_ORDER,
            edge_spacing=EDGE_SPACING,
        )

        logging.info('Predicting posterior...')
        results.append(
            classifier.predict(
                data["multiunit"].isel(time=test),
                time=data["position_info"].iloc[test].index,
            )
        )

        posterior = results[fold_ind].acausal_posterior.sum('state')

        trajectory_data = get_trajectory_data(
            posterior, track_graph, classifier,
            data['position_info'].iloc[test])

        error.append(
            np.abs(get_ahead_behind_distance(track_graph, *trajectory_data)))

        is_running = np.asarray(data["position_info"].iloc[test].speed > 4)

        error_info[f"median_error_fold_{fold_ind + 1}"] = np.nanmedian(
            error[fold_ind])
        error_info[f"run_median_error_fold_{fold_ind + 1}"] = np.nanmedian(
            error[fold_ind][is_running])

    results = (xr.concat(results, dim="time")
               .assign_coords(state=lambda ds:
                              ds.state.to_index().map(TRANSITION_TO_CATEGORY)))
    results["error"] = pd.Series(np.concatenate(error),
                                 index=data['position_info'].index)
    results['is_running'] = data['position_info'].speed > 4

    error_info['animal'] = animal
    error_info['day'] = int(day)
    error_info['epoch'] = int(epoch)

    epoch_identifier = f'{animal}_{day:02d}_{epoch:02d}_{data_type}_{dim}'
    prob = int(PROBABILITY_THRESHOLD * 100)
    error_info_filename = os.path.join(
        PROCESSED_DATA_DIR, f'{epoch_identifier}_error_info_{prob:02d}.csv')
    error_info.to_csv(error_info_filename)

    save_xarray(PROCESSED_DATA_DIR, epoch_key,
                results
                .drop(['likelihood', 'causal_posterior'])
                .sum('position'),
                group=f'/{data_type}/{dim}/classifier/run/')

    if plot_figures:
        logging.info('Plotting figures...')
        os.makedirs(os.path.join(FIGURE_DIR, 'run_decode'),  exist_ok=True)

        time_ind = list(range(0, len(results.time), 20_000))

        for slice_ind, (start_ind, end_ind) in enumerate(
                zip(time_ind[:-1], time_ind[1:])):
            time_slice = results.time[slice(start_ind, end_ind)]

            plot_classifier_time_slice(
                time_slice,
                classifier,
                results,
                data,
                posterior_type="acausal_posterior",
                figsize=(TWO_COLUMN, PAGE_HEIGHT * 0.8),
            )
            plt.suptitle(
                f'slice_ind = {animal}_{day:02d}_{epoch:02d}_'
                f'{slice_ind:02d}')
            fig_name = (
                f'{animal}_{day:02d}_{epoch:02d}_{slice_ind:02d}_'
                f'{data_type}_{dim}_acasual_classification.png')
            fig_name = os.path.join(FIGURE_DIR, 'run_decode', fig_name)
            plt.savefig(fig_name, bbox_inches='tight')
            plt.close(plt.gcf())

    logging.info('Done...\n')


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
    parser.add_argument('--plot_figures', action='store_true')
    parser.add_argument(
        '-d', '--debug',
        help='More verbose output for debugging',
        action='store_const',
        dest='log_level',
        const=logging.DEBUG,
        default=logging.INFO,
    )
    return parser.parse_args()


run_analysis = {
    ('clusterless', '1D'): clusterless_analysis_1D,
}


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
        epoch_key, plot_figures=args.plot_figures)


if __name__ == '__main__':
    sys.exit(main())
