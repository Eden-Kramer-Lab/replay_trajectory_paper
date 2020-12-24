'''Script for executing run_decode_all_times_by_epoch on the cluster
'''
from argparse import ArgumentParser
from os import environ, getcwd, makedirs
from os.path import join
from subprocess import run
from sys import exit

from loren_frank_data_processing import (make_epochs_dataframe,
                                         make_neuron_dataframe)
from src.parameters import _BRAIN_AREAS, ANIMALS


def get_command_line_arguments():
    parser = ArgumentParser()
    parser.add_argument('--Animal', type=str, help='Short name of animal')
    parser.add_argument('--Day', type=int, help='Day of recording session')
    parser.add_argument('--Epoch', type=int,
                        help='Epoch number of recording session')
    parser.add_argument('--data_type', type=str, default='clusterless')
    parser.add_argument('--dim', type=str, default='1D')
    parser.add_argument('--n_cores', type=int, default=16)
    parser.add_argument('--wall_time', type=str, default='36:00:00')
    parser.add_argument('--n_workers', type=int, default=16)
    parser.add_argument('--threads_per_worker', type=int, default=16)
    parser.add_argument('--plot_figures', action='store_true')

    return parser.parse_args()


def queue_job(python_cmd, directives=None, log_file='log.log',
              job_name='job'):
    queue_cmd = f'qsub {directives} -j y -o {log_file} -N {job_name}'
    cmd_line_script = f'echo python {python_cmd}  | {queue_cmd}'
    run(cmd_line_script, shell=True)


def main():

    args = get_command_line_arguments()

    # Set the maximum number of threads for openBLAS to use.
    num_threads = str(args.threads_per_worker)
    environ['OPENBLAS_NUM_THREADS'] = num_threads
    environ['MKL_NUM_THREADS'] = num_threads
    environ['NUMBA_NUM_THREADS'] = num_threads
    environ['OMP_NUM_THREADS'] = num_threads
    LOG_DIRECTORY = join(getcwd(), 'logs')
    makedirs(LOG_DIRECTORY,  exist_ok=True)

    python_function = 'run_decode_all_times_by_epoch.py'
    directives = ' '.join(
        [f'-l h_rt={args.wall_time}', f'-pe omp {args.n_cores}',
         '-P braincom', '-notify',
         '-v OPENBLAS_NUM_THREADS', '-v NUMBA_NUM_THREADS',
         '-v OMP_NUM_THREADS'])

    if args.Animal is None or args.Day is None or args.Epoch is None:
        epoch_info = make_epochs_dataframe(ANIMALS)
        neuron_info = make_neuron_dataframe(ANIMALS)
        neuron_info = neuron_info.loc[
            (neuron_info.type == 'principal') &
            (neuron_info.numspikes > 100) &
            neuron_info.area.isin(_BRAIN_AREAS)]
        n_neurons = (neuron_info
                     .groupby(['animal', 'day', 'epoch'])
                     .neuron_id
                     .agg(len)
                     .rename('n_neurons')
                     .to_frame())

        epoch_info = epoch_info.join(n_neurons)
        is_w_track = (epoch_info.environment
                      .isin(['TrackA', 'TrackB', 'WTrackA', 'WTrackB']))
        epoch_keys = epoch_info[is_w_track & (epoch_info.n_neurons > 9)].index
    else:
        epoch_keys = [(args.Animal, args.Day, args.Epoch)]

    for animal, day, epoch in epoch_keys:
        print(f'Animal: {animal}, Day: {day}, Epoch: {epoch}')

        log_file = (f'{animal}_{day:02d}_{epoch:02d}_{args.data_type}'
                    f'_{args.dim}.log')
        function_name = python_function.replace('.py', '')
        job_name = f'{function_name}_{animal}_{day:02d}_{epoch:02d}'
        python_cmd = (f'{python_function} {animal} {day} {epoch}'
                      f' --data_type {args.data_type}'
                      f' --dim {args.dim}'
                      f' --n_workers {args.n_workers}'
                      f' --threads_per_worker {args.threads_per_worker}')
        if args.plot_figures:
            python_cmd += ' --plot_figures'
        queue_job(python_cmd,
                  directives=directives,
                  log_file=join(LOG_DIRECTORY, log_file),
                  job_name=job_name)


if __name__ == '__main__':
    exit(main())
