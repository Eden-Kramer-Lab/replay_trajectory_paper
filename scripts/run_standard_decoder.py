import logging
import sys
from argparse import ArgumentParser
from signal import SIGUSR1, SIGUSR2, signal
from subprocess import PIPE, run

import xarray as xr
from loren_frank_data_processing import save_xarray
from src.parameters import PROCESSED_DATA_DIR
from src.standard_decoder import (fit_mark_likelihood, load_data,
                                  predict_clusterless_radon_wtrack)

FORMAT = '%(asctime)s %(message)s'

logging.basicConfig(level='INFO', format=FORMAT, datefmt='%d-%b-%y %H:%M:%S')


def clusterless_analysis_1D(epoch_key, dt=0.020):
    (
        linear_position,
        multiunit_dfs,
        _,
        ripple_times,
        track_graph,
        center_well_id,
    ) = load_data(epoch_key)
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

    results = []
    for ripple_number in ripple_times.index:
        (
            time,
            radon_estimated_velocity,
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
        results.append(
            xr.Dataset(
                data_vars={
                    "radon_prediction": (["time"], radon_prediction),
                    "radon_score": radon_score,
                    "radon_estimated_velocity": radon_estimated_velocity,
                },
                coords={"time": time + dt / 2},
            ))

    results = xr.concat(results, dim=ripple_times.index)
    save_xarray(PROCESSED_DATA_DIR,
                epoch_key,
                results,
                group='/clusterless/1D/standard_decoder/ripples/')


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
