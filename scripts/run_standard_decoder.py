import logging
import os
import sys
from argparse import ArgumentParser
from signal import SIGUSR1, SIGUSR2, signal
from subprocess import PIPE, run

import pandas as pd
from src.parameters import PROCESSED_DATA_DIR
from src.standard_decoder import (fit_mark_likelihood, load_data,
                                  predict_clusterless_radon_wtrack)

FORMAT = '%(asctime)s %(message)s'

logging.basicConfig(level='INFO', format=FORMAT, datefmt='%d-%b-%y %H:%M:%S')


def clusterless_analysis_1D(epoch_key, dt=0.020):
    logging.info('Loading data...')
    (
        linear_position,
        multiunit_dfs,
        _,
        ripple_times,
        track_graph,
        center_well_id,
    ) = load_data(epoch_key)
    logging.info('Fitting model...')
    (
        occupancy,
        joint_pdf_models,
        multiunit_dfs,
        ground_process_intensities,
        mean_rates,
        is_track_interior,
        place_bin_centers,
        place_bin_edges,
        edges,
    ) = fit_mark_likelihood(
        linear_position, multiunit_dfs, track_graph, center_well_id)

    logging.info('Predicting with radon...')
    radon_info = []
    for ripple_number in ripple_times.index:
        (
            time,
            radon_velocity,
            radon_prediction,
            radon_score,
            likelihood,
        ) = predict_clusterless_radon_wtrack(
            ripple_times,
            ripple_number,
            place_bin_centers,
            occupancy,
            joint_pdf_models,
            multiunit_dfs,
            ground_process_intensities,
            mean_rates,
            is_track_interior,
            place_bin_edges,
            dt=dt,
        )
        radon_info.append((radon_prediction[0], radon_velocity, radon_score))

    logging.info('Saving results...')
    radon_info = pd.DataFrame(
        radon_info,
        index=ripple_times.index,
        columns=["radon_start_position", "radon_velocity", "radon_score"],
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
    clusterless_analysis_1D(epoch_key)
    logging.info('Done...\n')


if __name__ == '__main__':
    sys.exit(main())
