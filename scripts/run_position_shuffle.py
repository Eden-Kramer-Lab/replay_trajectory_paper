import logging
import os
import sys
from argparse import ArgumentParser

import pandas as pd
import xarray as xr
from loren_frank_data_processing.position import (EDGE_ORDER, EDGE_SPACING,
                                                  make_track_graph)
from replay_trajectory_classification import ClusterlessClassifier
from scipy.ndimage import label
from src.analysis import get_replay_info
from src.load_data import load_data
from src.parameters import (ANIMALS, PROBABILITY_THRESHOLD, PROCESSED_DATA_DIR,
                            SAMPLING_FREQUENCY, TRANSITION_TO_CATEGORY)
from src.shuffle import shuffle_segments_run_position
from tqdm.auto import tqdm

FORMAT = "%(asctime)s %(message)s"

logging.basicConfig(level="INFO", format=FORMAT, datefmt="%d-%b-%y %H:%M:%S")


def classify(
    data, edge_order, edge_spacing, epoch_key, name="actual",
):
    is_training = ((data["position_info"].speed > 4) &
                   (data["position_info"].labeled_segments != 0))
    track_graph, center_well_id = make_track_graph(epoch_key, ANIMALS)

    animal, day, epoch = epoch_key
    data_type, dim = "clusterless", "1D"

    exclude_interneuron_spikes = False
    brain_areas = None

    # Set up naming
    group = f'/{data_type}/{dim}/'
    epoch_identifier = f'{animal}_{day:02d}_{epoch:02d}_{data_type}_{dim}'

    if exclude_interneuron_spikes:
        logging.info('Excluding interneuron spikes...')
        epoch_identifier += '_no_interneuron'
        group += 'no_interneuron/'

    if brain_areas is not None:
        area_str = '-'.join(brain_areas)
        epoch_identifier += f'_{area_str}'
        group += f'{area_str}/'

    model_name = os.path.join(
        PROCESSED_DATA_DIR, epoch_identifier + '_model.pkl')
    classifier = ClusterlessClassifier.load_model(model_name)
    classifier = shuffle_segments_run_position(
        data['position_info'], data['multiunit'], classifier)

    # Decode
    is_test = ~is_training

    test_groups = pd.DataFrame(
        {"test_groups": label(is_test.values)[0]}, index=is_test.index
    )
    immobility_results = []
    for _, df in test_groups.loc[is_test].groupby("test_groups"):
        start_time, end_time = df.iloc[0].name, df.iloc[-1].name
        test_multiunit = data["multiunit"].sel(
            time=slice(start_time, end_time))
        immobility_results.append(
            classifier.predict(test_multiunit, time=test_multiunit.time)
        )

    immobility_results = xr.concat(immobility_results, dim="time")

    results = [
        (
            immobility_results.sel(
                time=slice(df.start_time, df.end_time)
            ).assign_coords(time=lambda ds: ds.time - ds.time[0])
        )
        for _, df in data["ripple_times"].iterrows()
    ]

    results = xr.concat(results, dim=data["ripple_times"].index).assign_coords(
        state=lambda ds: ds.state.to_index().map(TRANSITION_TO_CATEGORY)
    )

    logging.info("Calculating replay_info...")
    spikes = (
        ((data["multiunit"].sum("features") > 0) * 1.0)
        .to_dataframe(name="spikes")
        .unstack()
    )
    spikes.columns = data["tetrode_info"].tetrode_id

    track_graph, _ = make_track_graph(epoch_key, ANIMALS)
    replay_info = get_replay_info(
        results,
        spikes,
        data["ripple_times"],
        data["position_info"],
        track_graph,
        SAMPLING_FREQUENCY,
        PROBABILITY_THRESHOLD,
        epoch_key,
        classifier,
        data["ripple_consensus_trace_zscore"],
    )

    logging.info("Saving replay_info...")
    epoch_identifier = f"{animal}_{day:02d}_{epoch:02d}_{data_type}_{dim}"
    os.makedirs(os.path.join(PROCESSED_DATA_DIR,
                             "run_position_shuffle"), exist_ok=True)
    replay_info_filename = os.path.join(
        PROCESSED_DATA_DIR,
        "run_position_shuffle",
        f"{epoch_identifier}_replay_info_{name}.csv",
    )
    replay_info.to_csv(replay_info_filename)


def get_command_line_arguments():
    parser = ArgumentParser()
    parser.add_argument('Animal', type=str, help='Short name of animal')
    parser.add_argument('Day', type=int, help='Day of recording session')
    parser.add_argument('Epoch', type=int,
                        help='Epoch number of recording session')
    parser.add_argument('--start_ind', type=int, default=0)
    parser.add_argument('--end_ind', type=int, default=50)

    return parser.parse_args()


def main():
    args = get_command_line_arguments()
    epoch_key = (args.Animal, args.Day, args.Epoch)
    logging.info(
        'Processing epoch: '
        f'Animal {args.Animal}, Day {args.Day}, Epoch #{args.Epoch}...')
    data = load_data(epoch_key)

    for shuffle_ind in tqdm(range(args.start_ind, args.end_ind),
                            desc='shuffle'):
        classify(
            data,
            EDGE_ORDER,
            EDGE_SPACING,
            epoch_key,
            name=f"run_position_shuffle_{shuffle_ind:02d}",
        )


if __name__ == '__main__':
    sys.exit(main())
