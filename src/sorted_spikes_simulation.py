import numpy as np

from replay_trajectory_classification.simulate import (
    simulate_linear_distance, simulate_neuron_with_place_field,
    simulate_place_field_firing_rate, simulate_time)

SAMPLING_FREQUENCY = 1000
TRACK_HEIGHT = 180
RUNNING_SPEED = 15
PLACE_FIELD_VARIANCE = 6.0 ** 2
PLACE_FIELD_MEANS = np.arange(0, TRACK_HEIGHT + 10, 10)
N_RUNS = 15
REPLAY_SPEEDUP = 120.0


def make_simulated_run_data(sampling_frequency=SAMPLING_FREQUENCY,
                            track_height=TRACK_HEIGHT,
                            running_speed=RUNNING_SPEED, n_runs=N_RUNS,
                            place_field_variance=PLACE_FIELD_VARIANCE,
                            place_field_means=PLACE_FIELD_MEANS):
    '''Make simulated data of a rat running back and forth
    on a linear maze with sorted spikes.

    Parameters
    ----------
    sampling_frequency : float, optional
    track_height : float, optional
    running_speed : float, optional
    n_runs : int, optional
    place_field_variance : float, optional
    place_field_means : ndarray, shape (n_neurons,), optional

    Returns
    -------
    time : ndarray, shape (n_time,)
    linear_distance : ndarray, shape (n_time,)
    sampling_frequency : float
    spikes : ndarray, shape (n_time, n_neurons)
    place_fields : ndarray, shape (n_time, n_neurons)

    '''
    n_samples = int(n_runs * sampling_frequency *
                    2 * track_height / running_speed)

    time = simulate_time(n_samples, sampling_frequency)
    linear_distance = simulate_linear_distance(
        time, track_height, running_speed)

    place_fields = np.stack(
        [simulate_place_field_firing_rate(place_field_mean, linear_distance,
                                          variance=place_field_variance)
         for place_field_mean in place_field_means], axis=1)

    spikes = np.stack([simulate_neuron_with_place_field(
        place_field_mean, linear_distance, max_rate=15,
        variance=place_field_variance, sampling_frequency=sampling_frequency)
        for place_field_mean in place_field_means.T], axis=1)

    return time, linear_distance, sampling_frequency, spikes, place_fields


def make_continuous_replay(sampling_frequency=SAMPLING_FREQUENCY,
                           track_height=TRACK_HEIGHT,
                           running_speed=RUNNING_SPEED,
                           place_field_means=PLACE_FIELD_MEANS,
                           replay_speedup=REPLAY_SPEEDUP):

    replay_speed = running_speed * replay_speedup
    n_samples = int(0.5 * sampling_frequency *
                    2 * track_height / replay_speed)
    replay_time = simulate_time(n_samples, sampling_frequency)
    true_replay_position = simulate_linear_distance(
        replay_time, track_height, replay_speed)

    min_times_ind = np.argmin(
        np.abs(true_replay_position[:, np.newaxis] - place_field_means),
        axis=0)

    n_neurons = place_field_means.shape[0]
    test_spikes = np.zeros((replay_time.size, n_neurons))
    test_spikes[(min_times_ind, np.arange(n_neurons),)] = 1.0

    return replay_time, test_spikes


def make_hover_replay(hover_neuron_ind=None,
                      place_field_means=PLACE_FIELD_MEANS,
                      sampling_frequency=SAMPLING_FREQUENCY):

    n_neurons = place_field_means.shape[0]
    if hover_neuron_ind is None:
        hover_neuron_ind = n_neurons // 2

    N_TIME = 50
    replay_time = np.arange(N_TIME) / sampling_frequency

    spike_time_ind = np.arange(0, N_TIME, 2)

    test_spikes = np.zeros((N_TIME, n_neurons))
    neuron_ind = np.ones((N_TIME // 2,), dtype=np.int) * hover_neuron_ind

    test_spikes[(spike_time_ind, neuron_ind)] = 1.0

    return replay_time, test_spikes


def make_fragmented_replay(place_field_means=PLACE_FIELD_MEANS,
                           sampling_frequency=SAMPLING_FREQUENCY):
    N_TIME = 10
    replay_time = np.arange(N_TIME) / sampling_frequency
    ind = ([1, 3, 5, 7, 9], [1, -1, 10, -5, 8])
    n_neurons = place_field_means.shape[0]
    test_spikes = np.zeros((N_TIME, n_neurons))
    test_spikes[ind] = 1.0

    return replay_time, test_spikes


def make_hover_continuous_hover_replay(sampling_frequency=SAMPLING_FREQUENCY):
    _, test_spikes1 = make_hover_replay(hover_neuron_ind=0)
    _, test_spikes2 = make_continuous_replay()
    _, test_spikes3 = make_hover_replay(hover_neuron_ind=-1)

    test_spikes = np.concatenate((test_spikes1, test_spikes2, test_spikes3))
    replay_time = np.arange(test_spikes.shape[0]) / sampling_frequency

    return replay_time, test_spikes


def make_fragmented_hover_fragmented_replay(
        sampling_frequency=SAMPLING_FREQUENCY):
    _, test_spikes1 = make_fragmented_replay()
    _, test_spikes2 = make_hover_replay(hover_neuron_ind=6)
    _, test_spikes3 = make_fragmented_replay()

    test_spikes = np.concatenate((test_spikes1, test_spikes2, test_spikes3))
    replay_time = np.arange(test_spikes.shape[0]) / sampling_frequency

    return replay_time, test_spikes


def make_fragmented_continuous_fragmented_replay(
        sampling_frequency=SAMPLING_FREQUENCY):
    _, test_spikes1 = make_fragmented_replay()
    _, test_spikes2 = make_continuous_replay()
    _, test_spikes3 = make_fragmented_replay()

    test_spikes = np.concatenate((test_spikes1, test_spikes2, test_spikes3))
    replay_time = np.arange(test_spikes.shape[0]) / sampling_frequency

    return replay_time, test_spikes
