import logging
import os
import sys
from argparse import ArgumentParser
from signal import SIGUSR1, SIGUSR2, signal
from subprocess import PIPE, run

import dask
import numpy as np
import pandas as pd
from dask.distributed import Client
from src.parameters import PROCESSED_DATA_DIR
from src.shuffle import get_shuffled_pvalue, shuffle_likelihood_position_bins
from src.standard_decoder import (fit_mark_likelihood, load_data,
                                  predict_clusterless_wtrack,
                                  predict_mark_likelihood)

FORMAT = '%(asctime)s %(message)s'

logging.basicConfig(level='INFO', format=FORMAT, datefmt='%d-%b-%y %H:%M:%S')


@dask.delayed
def get_shuffled_scores(
    time,
    likelihood,
    is_track_interior,
    place_bin_centers,
    place_bin_edges,
    track_graph1,
    place_bin_center_ind_to_node,
    dt,
):
    shuffled_likelihood = shuffle_likelihood_position_bins(
        likelihood, is_track_interior
    )
    (
        _,
        _,
        _,
        _,
        shuffled_radon_score,
        _,
        _,
        shuffled_isotonic_score,
        _,
        _,
        shuffled_linear_score,
        _,
        _,
        shuffled_map_score,
    ) = predict_clusterless_wtrack(
        time,
        shuffled_likelihood,
        place_bin_centers,
        is_track_interior,
        place_bin_edges,
        track_graph1,
        place_bin_center_ind_to_node,
        dt=dt,
    )

    return (
        shuffled_radon_score,
        shuffled_isotonic_score,
        shuffled_linear_score,
        shuffled_map_score,
    )


def clusterless_analysis_1D(epoch_key, dt=0.020, n_shuffles=1000):
    logging.info('Loading data...')
    (
        linear_position,
        multiunit_dfs,
        _,
        ripple_times,
        track_graph,
        center_well_id,
        _,
    ) = load_data(epoch_key)
    logging.info('Fitting model...')
    (
        occupancy,
        joint_pdf_models,
        multiunit_dfs,
        ground_process_intensities,
        mean_rates,
        place_bin_centers,
        place_bin_edges,
        is_track_interior,
        distance_between_nodes,
        place_bin_center_ind_to_node,
        place_bin_center_2D_position,
        place_bin_edges_2D_position,
        centers_shape,
        edges,
        track_graph1,
        place_bin_center_ind_to_edge_id,
        nodes_df,
    ) = fit_mark_likelihood(
        linear_position, multiunit_dfs, track_graph, center_well_id)

    logging.info('Predicting with radon...')
    radon_info = []
    for ripple_number in ripple_times.index:
        start_time, end_time = (
            ripple_times.loc[ripple_number].start_time /
            np.timedelta64(1, "s"),
            ripple_times.loc[ripple_number].end_time / np.timedelta64(1, "s"),
        )

        likelihood, time = predict_mark_likelihood(
            start_time,
            end_time,
            place_bin_centers,
            occupancy,
            joint_pdf_models,
            multiunit_dfs,
            ground_process_intensities,
            mean_rates,
            is_track_interior,
            dt=dt,
        )
        (_, _,
            radon_speed, _, radon_score,
            isotonic_speed, _, isotonic_score,
            linear_speed, _, linear_score,
            map_speed, _, map_score,
         ) = predict_clusterless_wtrack(
            time,
            likelihood,
            place_bin_centers,
            is_track_interior,
            place_bin_edges,
            track_graph1,
            place_bin_center_ind_to_node,
            dt=dt,
        )

        scores = []

        for shuffle_ind in range(n_shuffles):
            scores.append(
                get_shuffled_scores(
                    time,
                    likelihood,
                    is_track_interior,
                    place_bin_centers,
                    place_bin_edges,
                    track_graph1,
                    place_bin_center_ind_to_node,
                    dt,
                )
            )

        (shuffled_radon_score, shuffled_isotonic_score, shuffled_linear_score,
         shuffled_map_score) = np.asarray(dask.compute(*scores)).T

        radon_pvalue = get_shuffled_pvalue(radon_score, shuffled_radon_score)
        isotonic_pvalue = get_shuffled_pvalue(
            isotonic_score, shuffled_isotonic_score)
        linear_pvalue = get_shuffled_pvalue(
            linear_score, shuffled_linear_score)
        map_pvalue = get_shuffled_pvalue(map_score, shuffled_map_score)

        radon_info.append(
            (radon_speed, radon_score, radon_pvalue,
             isotonic_speed, isotonic_score, isotonic_pvalue,
             linear_speed, linear_score, linear_pvalue,
             map_speed, map_score, map_pvalue))

    logging.info('Saving results...')
    radon_info = pd.DataFrame(
        radon_info,
        index=ripple_times.index,
        columns=["radon_speed", "radon_score", "radon_pvalue",
                 "isotonic_speed", "isotonic_score", "isotonic_pvalue",
                 "linear_speed", "linear_score", "linear_pvalue",
                 "map_speed", "map_score", "map_pvalue"],
    )

    animal, day, epoch = epoch_key
    radon_info["animal"] = animal
    radon_info["day"] = int(day)
    radon_info["epoch"] = int(epoch)

    radon_info_filename = os.path.join(
        PROCESSED_DATA_DIR, f'{animal}_{day:02d}_{epoch:02d}_radon_info.csv')
    radon_info.to_csv(radon_info_filename)


def get_command_line_arguments():
    parser = ArgumentParser()
    parser.add_argument('Animal', type=str, help='Short name of animal')
    parser.add_argument('Day', type=int, help='Day of recording session')
    parser.add_argument('Epoch', type=int,
                        help='Epoch number of recording session')
    parser.add_argument('--n_workers', type=int, default=16)
    parser.add_argument('--threads_per_worker', type=int, default=1)
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
        logging.error(f'***Process killed with signal {signal_code}***')
        exit()

    for code in [SIGUSR1, SIGUSR2]:
        signal(code, _signal_handler)

    epoch_key = (args.Animal, args.Day, args.Epoch)
    logging.info(
        'Processing epoch: Animal {0}, Day {1}, Epoch #{2}...'.format(
            *epoch_key))
    logging.info('Standard Decoder')
    git_hash = run(['git', 'rev-parse', 'HEAD'],
                   stdout=PIPE, universal_newlines=True).stdout
    logging.info('Git Hash: {git_hash}'.format(git_hash=git_hash.rstrip()))

    # Analysis Code
    client = Client(processes=True, threads_per_worker=args.threads_per_worker,
                    n_workers=args.n_workers, memory_limit="32GB")
    with client:
        clusterless_analysis_1D(epoch_key)
    logging.info('Done...\n')


if __name__ == '__main__':
    sys.exit(main())
