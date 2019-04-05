import networkx as nx
import numpy as np
import pandas as pd
from loren_frank_data_processing.position import (get_interpolated_position_dataframe,
                                                  get_trial_time,
                                                  make_track_graph)
from replay_trajectory_classification.simulate import \
    simulate_neuron_with_place_field

from src.parameters import ANIMALS, SAMPLING_FREQUENCY


def _get_inbetween_nodes(node1, node2, node_positions, position_ind, n_nodes):
    node_spacing = (node_positions[node2][position_ind] -
                    node_positions[node1][position_ind]) / (n_nodes + 1)
    inbetween_positions = (node_positions[node1][position_ind] +
                           np.arange(1, n_nodes + 1) * node_spacing)
    inbetween_positions = inbetween_positions[:, np.newaxis]
    other_position_ind = int(position_ind == 0)

    if other_position_ind == 0:
        x = (np.ones_like(inbetween_positions) *
             node_positions[node1][other_position_ind])
        return np.concatenate((x, inbetween_positions), axis=1)
    else:
        y = (np.ones_like(inbetween_positions) *
             node_positions[node1][other_position_ind])
        return np.concatenate((inbetween_positions, y), axis=1)


def _get_euclidean_dist(x):
    return np.linalg.norm(x[:-1] - x[1:], axis=1)


def load_simulated_spikes_with_real_position():
    epoch_key = ('bon', 3, 2)

    track_graph, _ = make_track_graph(epoch_key, ANIMALS)

    time = get_trial_time(epoch_key, ANIMALS)
    time = (pd.Series(np.ones_like(time, dtype=np.float), index=time)
            .resample('1ms').mean()
            .index)

    def _time_function(*args, **kwargs):
        return time

    position_info = (
        get_interpolated_position_dataframe(epoch_key, ANIMALS, _time_function)
        .dropna(subset=['linear_distance', 'linear_speed']))

    node_positions = nx.get_node_attributes(track_graph, 'pos')

    place_field_centers = np.concatenate((
        np.asarray(list(node_positions.values())),
        _get_inbetween_nodes(2, 3, node_positions, position_ind=1, n_nodes=2),
        _get_inbetween_nodes(1, 0, node_positions, position_ind=1, n_nodes=2),
        _get_inbetween_nodes(4, 5, node_positions, position_ind=1, n_nodes=2),
        _get_inbetween_nodes(1, 2, node_positions, position_ind=0, n_nodes=1),
        _get_inbetween_nodes(4, 1, node_positions, position_ind=0, n_nodes=1),
    ), axis=0)

    position = position_info.loc[:, ['x_position', 'y_position']].values
    is_training = position_info.speed > 4
    spikes = np.stack(
        [simulate_neuron_with_place_field(
            center, position, max_rate=15, sigma=100,
            sampling_frequency=SAMPLING_FREQUENCY)
         for center in place_field_centers], axis=1)

    return position, spikes, is_training, place_field_centers, position_info


def continuous_replay(place_field_centers, replay_speed=1200.0):
    '''
    Parameters
    ----------
    place_field_centers : ndarray, shape (n_neurons, 2)
    replay_speed : float, optional
        Centimeters per second a replay can travel
    '''
    neuron_ind = np.asarray([0, 9, 8, 1, 13, 4, 10, 11, 5])
    dist = _get_euclidean_dist(place_field_centers[neuron_ind])

    spike_time_ind = np.cumsum(
        np.insert(np.ceil(SAMPLING_FREQUENCY * dist / replay_speed), 0, 2)
    ).astype(np.int)

    n_neurons = place_field_centers.shape[0]
    n_time = int(spike_time_ind.max() + 1)
    time = np.arange(n_time) / SAMPLING_FREQUENCY
    test_spikes = np.zeros((n_time, n_neurons))
    test_spikes[(spike_time_ind, neuron_ind)] = 1.0

    return test_spikes, time


def hover_replay(place_field_centers):
    neuron_ind = np.zeros((6,), dtype=np.int)

    spike_time_ind = np.arange(0, neuron_ind.size * 5, 5)

    n_neurons = place_field_centers.shape[0]
    n_time = int(spike_time_ind.max() + 1)
    time = np.arange(n_time) / SAMPLING_FREQUENCY
    test_spikes = np.zeros((n_time, n_neurons))
    test_spikes[(spike_time_ind, neuron_ind)] = 1.0

    return test_spikes, time


def fragmented_replay(place_field_centers):
    neuron_ind = np.asarray([0, 11, 8, 4, 2, 10, 9, 5, 3])
    spike_time_ind = np.arange(0, neuron_ind.size * 5, 5)
    n_neurons = place_field_centers.shape[0]
    n_time = int(spike_time_ind.max() + 1)
    time = np.arange(n_time) / SAMPLING_FREQUENCY
    test_spikes = np.zeros((n_time, n_neurons))
    test_spikes[(spike_time_ind, neuron_ind)] = 1.0

    return test_spikes, time


def hover_continuous_hover_replay(place_field_centers):
    neuron_ind = np.asarray([0, 0, 0, 0, 0, 0,
                             0, 9, 8, 1,
                             1, 1, 1, 1, 1, 1])
    spike_time_ind = np.asarray([0, 5, 10, 15, 20, 25,
                                 30, 51, 72, 93,
                                 98, 102, 107, 112, 117, 123])
    n_neurons = place_field_centers.shape[0]
    n_time = int(spike_time_ind.max() + 1)
    time = np.arange(n_time) / SAMPLING_FREQUENCY
    test_spikes = np.zeros((n_time, n_neurons))
    test_spikes[(spike_time_ind, neuron_ind)] = 1.0

    return test_spikes, time


def continuous_fragmented_continuous_replay(place_field_centers):
    neuron_ind = np.asarray([3, 7, 6, 2,
                             5, 0, 6, 4,
                             0, 9, 8, 1])
    spike_time_ind = np.asarray([0, 21, 42, 63,
                                 68, 72, 77, 82,
                                 85, 106, 127, 148])
    n_neurons = place_field_centers.shape[0]
    n_time = int(spike_time_ind.max() + 1)
    time = np.arange(n_time) / SAMPLING_FREQUENCY
    test_spikes = np.zeros((n_time, n_neurons))
    test_spikes[(spike_time_ind, neuron_ind)] = 1.0

    return test_spikes, time


def hover_fragmented_hover_replay(place_field_centers):
    neuron_ind = np.asarray([0, 0, 0, 0, 0, 0,
                             4, 9, 10, 2,
                             3, 3, 3, 3, 3, 3])
    spike_time_ind = np.asarray([5, 10, 15, 20, 25, 30,
                                 35, 40, 45, 50,
                                 55, 60, 65, 70, 75, 80])
    n_neurons = place_field_centers.shape[0]
    n_time = int(spike_time_ind.max() + 1)
    time = np.arange(n_time) / SAMPLING_FREQUENCY
    test_spikes = np.zeros((n_time, n_neurons))
    test_spikes[(spike_time_ind, neuron_ind)] = 1.0

    return test_spikes, time
