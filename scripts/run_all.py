import os
import subprocess
import sys

from tqdm.auto import tqdm

from loren_frank_data_processing import (make_epochs_dataframe,
                                         make_neuron_dataframe)
from src.parameters import (ANIMALS, MIN_N_NEURONS, MAX_N_EXPOSURES,
                            PROCESSED_DATA_DIR)


def run_bash(epoch_key, log_directory):
    animal, day, epoch = epoch_key
    print(f'Animal: {animal}, Day: {day}, Epoch: {epoch}')
    bash_cmd = f'python run_by_epoch.py {animal} {day} {epoch}'
    log_file = os.path.join(
        log_directory, f'{animal}_{day:02d}_{epoch:02d}.log')
    with open(log_file, 'w') as f:
        try:
            subprocess.run(bash_cmd, shell=True, check=True,
                           stderr=subprocess.STDOUT, stdout=f)
        except subprocess.CalledProcessError:
            print(f'Error in {epoch_key}')


def main():
    epoch_info = make_epochs_dataframe(ANIMALS)
    neuron_info = make_neuron_dataframe(ANIMALS)
    n_neurons = (neuron_info
                 .groupby(['animal', 'day', 'epoch'])
                 .neuron_id
                 .agg(len)
                 .rename('n_neurons')
                 .to_frame())

    epoch_info = epoch_info.join(n_neurons)
    is_w_track = (epoch_info.environment
                  .isin(['TrackA', 'TrackB', 'WTrackA', 'WTrackB']))
    is_animal = epoch_info.index.isin(
        ['bon', 'fra', 'gov', 'dud', 'con'], level='animal')
    valid_epochs = (is_w_track &
                    (epoch_info.n_neurons > MIN_N_NEURONS) &
                    (epoch_info.exposure <= MAX_N_EXPOSURES) &
                    is_animal
                    )
    log_directory = os.path.join(os.getcwd(), 'logs')
    os.makedirs(log_directory,  exist_ok=True)

    for epoch_key in tqdm(epoch_info[valid_epochs].index, desc='epochs'):
        animal, day, epoch = epoch_key
        replay_info_filename = os.path.join(
            PROCESSED_DATA_DIR,
            f'{animal}_{day:02d}_{epoch:02d}_sorted_spikes_replay_info.csv')
        if ~os.path.isfile(replay_info_filename):
            run_bash(epoch_key, log_directory)


if __name__ == '__main__':
    sys.exit(main())
