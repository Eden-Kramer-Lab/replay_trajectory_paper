import subprocess
import sys

from loren_frank_data_processing import (make_epochs_dataframe,
                                         make_neuron_dataframe)
from src.parameters import ANIMALS


def run_bash(epoch_keys):
    bash_cmd = [f'python run_by_epoch.py {animal} {day} {epoch}'
                for animal, day, epoch in epoch_keys]
    bash_cmd = '; '.join(bash_cmd)
    with open('logs/log.log', 'w') as f:
        subprocess.run(bash_cmd, shell=True, check=True,
                       stderr=subprocess.STDOUT, stdout=f)


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
                    (epoch_info.n_neurons > 20) &
                    (epoch_info.exposure < 4) &
                    is_animal
                    )
    run_bash(epoch_info[valid_epochs].index)


if __name__ == '__main__':
    sys.exit(main())
