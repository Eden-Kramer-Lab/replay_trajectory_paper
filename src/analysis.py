import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr
from loren_frank_data_processing.track_segment_classification import (
    get_track_segments_from_graph, project_points_to_segment)


def get_replay_info(results, ripple_spikes, ripple_times, position_info,
                    track_graph, sampling_frequency, probability_threshold,
                    epoch_key, classifier):
    '''

    Parameters
    ----------
    results : xarray.Dataset, shape (n_ripples, n_position_bins, n_states,
                                     n_ripple_time)
    ripple_spikes : pandas.DataFrame (n_ripples * n_ripple_time, n_neurons)
    ripple_times : pandas.DataFrame (n_ripples, 2)
    position_info : pandas.DataFrame (n_time, n_covariates)
    track_graph : networkx.Graph
    sampling_frequency : float
    probability_threshold : float
    epoch_key : tuple

    Returns
    -------
    replay_info : pandas.DataFrame, shape (n_ripples, n_covariates)

    '''
    probability = get_probability(results)
    is_classified = get_is_classified(probability, probability_threshold)

    duration = (is_classified.sum('time') / sampling_frequency)
    duration = duration.to_dataframe().unstack(level=1)
    duration.columns = list(duration.columns.get_level_values('state'))
    duration = duration.rename(
        columns=lambda column_name: column_name + '_duration')
    is_category = (duration > 0.0).rename(columns=lambda c: c.split('_')[0])
    duration = pd.concat((duration, is_category), axis=1)
    duration['is_classified'] = np.any(duration > 0.0, axis=1)
    duration['n_unique_spiking'] = get_n_unique_spiking(ripple_spikes)
    duration['n_total_spikes'] = get_n_total_spikes(ripple_spikes)

    ripple_position_info = reshape_to_segments(position_info, ripple_times)
    duration['actual_x_position'] = ripple_position_info.groupby(
        'ripple_number').x_position.mean()
    duration['actual_y_position'] = ripple_position_info.groupby(
        'ripple_number').y_position.mean()
    duration['actual_linear_distance'] = ripple_position_info.groupby(
        'ripple_number').linear_distance.mean()
    duration['actual_linear_position'] = ripple_position_info.groupby(
        'ripple_number').linear_position.mean()
    duration['actual_speed'] = ripple_position_info.groupby(
        'ripple_number').speed.mean()
    duration['actual_velocity_center_well'] = ripple_position_info.groupby(
        'ripple_number').linear_velocity.mean()

    metrics = pd.DataFrame(
        [get_replay_distance_metrics(
            results, ripple_position_info, ripple_spikes, ripple_number,
            position_info, track_graph, sampling_frequency, is_classified,
            probability)
         for ripple_number in ripple_times.index], index=ripple_times.index)
    replay_info = pd.concat((ripple_times, duration, metrics), axis=1)
    animal, day, epoch = epoch_key

    replay_info['animal'] = animal
    replay_info['day'] = int(day)
    replay_info['epoch'] = int(epoch)

    min_max = (
        classifier
        ._nodes_df[classifier._nodes_df.is_bin_edge]
        .groupby('edge_id')
        .aggregate(['min', 'max']))

    replay_info['center_well_position'] = min_max.loc[0].linear_position.min()
    replay_info['choice_position'] = min_max.loc[0].linear_position.max()

    replay_info['left_arm_start'] = min_max.loc[1].linear_position.min()
    replay_info['left_well_position'] = min_max.loc[3].linear_position.max()

    replay_info['right_arm_start'] = min_max.loc[2].linear_position.min()
    replay_info['right_well_position'] = min_max.loc[4].linear_position.max()
    center_well_id = 0
    replay_info['max_linear_distance'] = list(
        classifier.distance_between_nodes_[center_well_id].values())[-1]

    return replay_info


def get_sleep_replay_info(results, ripple_spikes, ripple_times, position_info,
                          sampling_frequency, probability_threshold,
                          epoch_key, classifier):
    '''

    Parameters
    ----------
    results : xarray.Dataset, shape (n_ripples, n_position_bins, n_states,
                                     n_ripple_time)
    ripple_spikes : pandas.DataFrame (n_ripples * n_ripple_time, n_neurons)
    ripple_times : pandas.DataFrame (n_ripples, 2)
    position_info : pandas.DataFrame (n_time, n_covariates)
    sampling_frequency : float
    probability_threshold : float
    epoch_key : tuple

    Returns
    -------
    replay_info : pandas.DataFrame, shape (n_ripples, n_covariates)

    '''
    probability = get_probability(results)
    is_classified = get_is_classified(probability, probability_threshold)

    duration = (is_classified.sum('time') / sampling_frequency)
    duration = duration.to_dataframe().unstack(level=1)
    duration.columns = list(duration.columns.get_level_values('state'))
    duration = duration.rename(
        columns=lambda column_name: column_name + '_duration')
    is_category = (duration > 0.0).rename(columns=lambda c: c.split('_')[0])
    duration = pd.concat((duration, is_category), axis=1)
    duration['is_classified'] = np.any(duration > 0.0, axis=1)
    duration['n_unique_spiking'] = get_n_unique_spiking(ripple_spikes)
    duration['n_total_spikes'] = get_n_total_spikes(ripple_spikes)

    ripple_position_info = reshape_to_segments(position_info, ripple_times)
    duration['actual_x_position'] = ripple_position_info.groupby(
        'ripple_number').x_position.mean()
    duration['actual_y_position'] = ripple_position_info.groupby(
        'ripple_number').y_position.mean()
    duration['actual_speed'] = ripple_position_info.groupby(
        'ripple_number').speed.mean()

    replay_info = pd.concat((ripple_times, duration), axis=1)
    animal, day, epoch = epoch_key

    replay_info['animal'] = animal
    replay_info['day'] = int(day)
    replay_info['epoch'] = int(epoch)

    min_max = (
        classifier
        ._nodes_df[classifier._nodes_df.is_bin_edge]
        .groupby('edge_id')
        .aggregate(['min', 'max']))

    replay_info['center_well_position'] = min_max.loc[0].linear_position.min()
    replay_info['choice_position'] = min_max.loc[0].linear_position.max()

    replay_info['left_arm_start'] = min_max.loc[1].linear_position.min()
    replay_info['left_well_position'] = min_max.loc[3].linear_position.max()

    replay_info['right_arm_start'] = min_max.loc[2].linear_position.min()
    replay_info['right_well_position'] = min_max.loc[4].linear_position.max()
    center_well_id = 0
    replay_info['max_linear_distance'] = list(
        classifier.distance_between_nodes_[center_well_id].values())[-1]

    return replay_info


def get_probability(results):
    '''Get probability of each state and two states derived from mixtures of
    each state.

    Parameters
    ----------
    results : xarray.Dataset

    Returns
    -------
    probability : xarray.DataArray

    '''
    try:
        probability = (results
                       .acausal_posterior
                       .sum(['x_position', 'y_position'], skipna=True))
    except ValueError:
        probability = (results
                       .acausal_posterior
                       .dropna('position', how='all')
                       .sum('position', skipna=False))

    return xr.concat(
        (probability,
         probability
            .sel(state=['Hover', 'Continuous'])
            .sum('state', skipna=False)
            .assign_coords(state='Hover-Continuous-Mix'),
         probability
            .sel(state=['Fragmented', 'Continuous'])
            .sum('state', skipna=False)
            .assign_coords(state='Fragmented-Continuous-Mix'),
         ), dim='state')


def get_is_classified(probability, probablity_threshold):
    '''Classify each state by the confidence threshold and make sure two
    derived states exclude their parent states.

    Parameters
    ----------
    probability : xarray.DataArray
    probablity_threshold : float

    Returns
    -------
    is_classified : xarray.DataArray

    '''
    is_classified = probability > probablity_threshold
    is_classified.loc[dict(state='Hover-Continuous-Mix')] = (
        is_classified.sel(state='Hover-Continuous-Mix') &
        ~is_classified.sel(state='Hover') &
        ~is_classified.sel(state='Continuous') &
        (probability.sel(state='Fragmented') < (1 - probablity_threshold) / 2))

    is_classified.loc[dict(state='Fragmented-Continuous-Mix')] = (
        is_classified.sel(state='Fragmented-Continuous-Mix') &
        ~is_classified.sel(state='Fragmented') &
        ~is_classified.sel(state='Continuous') &
        (probability.sel(state='Hover') < (1 - probablity_threshold) / 2))
    is_classified = is_classified.rename('is_classified')
    is_classified = is_classified.where(~np.isnan(probability))

    return is_classified


def get_replay_distance_metrics(results, ripple_position_info, ripple_spikes,
                                ripple_number, position_info, track_graph,
                                sampling_frequency, is_classified,
                                probability):
    posterior = (results
                 .sel(ripple_number=ripple_number)
                 .acausal_posterior
                 .dropna('time', how='all')
                 .assign_coords(
                     time=lambda ds: 1000 * ds.time / np.timedelta64(1, 's')))
    is_classified = (
        is_classified
        .sel(ripple_number=ripple_number)
        .dropna('time', how='all')
        .assign_coords(
            time=lambda ds: 1000 * ds.time / np.timedelta64(1, 's')))
    probability = (
        probability
        .sel(ripple_number=ripple_number)
        .dropna('time', how='all')
        .assign_coords(
            time=lambda ds: 1000 * ds.time / np.timedelta64(1, 's'))
    )
    ripple_spikes = ripple_spikes.loc[ripple_number]
    map_estimate = maximum_a_posteriori_estimate(posterior.sum('state'))

    actual_positions = (ripple_position_info
                        .loc[ripple_number, ['x_position', 'y_position']]
                        .values)
    actual_track_segment_ids = (ripple_position_info
                                .loc[ripple_number, 'track_segment_id']
                                .values.squeeze().astype(int))

    (replay_distance_from_actual_position,
     replay_distance_from_center_well) = calculate_replay_distance(
        track_graph, map_estimate, actual_positions,
        actual_track_segment_ids, position_info)
    try:
        replay_total_displacement = np.abs(
            replay_distance_from_actual_position[-1] -
            replay_distance_from_actual_position[0])
    except IndexError:
        replay_total_displacement = np.nan

    map_estimate = map_estimate.squeeze()

    metrics = {
        'replay_distance_from_actual_position': np.mean(
            replay_distance_from_actual_position),
        'replay_speed': np.mean(
            np.abs(np.diff(replay_distance_from_actual_position)) /
            sampling_frequency),
        'replay_velocity_actual_position': np.mean(
            np.diff(replay_distance_from_actual_position) /
            sampling_frequency),
        'replay_velocity_center_well': np.mean(
            np.diff(replay_distance_from_center_well)
            / sampling_frequency),
        'replay_distance_from_center_well': np.mean(
            replay_distance_from_center_well),
        'replay_linear_position': np.mean(map_estimate),
        'replay_total_distance': np.sum(
            np.abs(np.diff(replay_distance_from_actual_position))),
        'replay_total_displacement': replay_total_displacement,
    }

    for state, above_threshold in is_classified.groupby('state'):
        above_threshold = above_threshold.astype(bool).values.squeeze()
        try:
            metrics[f'{state}_max_probability'] = np.max(
                np.asarray(probability.sel(state=state)))
        except ValueError:
            metrics[f'{state}_max_probability'] = np.nan
        if np.any(above_threshold):
            time = np.asarray(posterior.time)
            state_distance = replay_distance_from_actual_position[
                above_threshold]

            metrics[f'{state}_replay_distance_from_actual_position'] = np.mean(
                state_distance)
            metrics[f'{state}_replay_speed'] = np.mean(
                np.abs(np.diff(state_distance)) / sampling_frequency)
            metrics[f'{state}_replay_velocity_actual_position'] = np.mean(
                np.diff(state_distance) / sampling_frequency)
            metrics[f'{state}_replay_velocity_center_well'] = np.mean(
                np.diff(replay_distance_from_center_well[above_threshold])
                / sampling_frequency)
            metrics[f'{state}_replay_distance_from_center_well'] = np.mean(
                replay_distance_from_center_well[above_threshold])
            metrics[f'{state}_replay_linear_position'] = np.mean(
                map_estimate[above_threshold])
            metrics[f'{state}_replay_total_distance'] = np.sum(
                np.abs(np.diff(state_distance)))
            metrics[f'{state}_replay_total_displacement'] = np.abs(
                state_distance[-1] - state_distance[0])
            metrics[f'{state}_min_time'] = np.min(time[above_threshold])
            metrics[f'{state}_max_time'] = np.max(time[above_threshold])
            metrics[f'{state}_n_unique_spiking'] = (
                ripple_spikes.sum(axis=0) > 0).sum()
            metrics[f'{state}_n_total_spikes'] = (
                ripple_spikes.sum(axis=0)).sum()
            metrics[f'{state}_population_rate'] = (
                sampling_frequency * metrics[f'{state}_n_total_spikes'] /
                above_threshold.sum())

    return metrics


def get_n_unique_spiking(ripple_spikes):
    return (ripple_spikes.groupby('ripple_number').sum() > 0).sum(axis=1)


def get_n_total_spikes(ripple_spikes):
    return ripple_spikes.groupby('ripple_number').sum().sum(axis=1)


def maximum_a_posteriori_estimate(posterior_density):
    '''

    Parameters
    ----------
    posterior_density : xarray.DataArray, shape (n_time, n_x_bins, n_y_bins)

    Returns
    -------
    map_estimate : ndarray, shape (n_time,)

    '''
    try:
        stacked_posterior = np.log(posterior_density.stack(
            z=['x_position', 'y_position']))
        map_estimate = stacked_posterior.z[stacked_posterior.argmax('z')]
        map_estimate = np.asarray(map_estimate.values.tolist())
    except KeyError:
        map_estimate = posterior_density.position[
            np.log(posterior_density).argmax('position')]
        map_estimate = np.asarray(map_estimate)[:, np.newaxis]
    return map_estimate


def get_place_field_max(classifier):
    try:
        max_ind = classifier.place_fields_.argmax('position')
        return np.asarray(
            classifier.place_fields_.position[max_ind].values.tolist())
    except AttributeError:
        return np.asarray(
            [classifier.place_bin_centers_[gpi.argmax()]
             for gpi in classifier.ground_process_intensities_])


def get_linear_position_order(position_info, place_field_max):
    position = position_info.loc[:, ['x_position', 'y_position']]
    linear_place_field_max = []

    for place_max in place_field_max:
        min_ind = np.sqrt(
            np.sum(np.abs(place_max - position) ** 2, axis=1)).idxmin()
        linear_place_field_max.append(
            position_info.loc[min_ind, 'linear_position'])

    linear_place_field_max = np.asarray(linear_place_field_max)
    return np.argsort(linear_place_field_max), linear_place_field_max


def reshape_to_segments(time_series, segments):
    df = []
    for row in segments.itertuples():
        row_series = time_series.loc[row.start_time:row.end_time]
        row_series.index = row_series.index - row_series.index[0]
        df.append(row_series)

    return pd.concat(df, axis=0, keys=segments.index).sort_index()


def _get_closest_ind(map_estimate, all_positions):
    map_estimate = np.asarray(map_estimate)
    all_positions = np.asarray(all_positions)
    return np.argmin(np.linalg.norm(
        map_estimate[:, np.newaxis, :] - all_positions[np.newaxis, ...],
        axis=-2), axis=1)


def _get_projected_track_positions(position, track_segments, track_segment_id):
    projected_track_positions = project_points_to_segment(
        track_segments, position)
    n_time = projected_track_positions.shape[0]
    projected_track_positions = projected_track_positions[(
        np.arange(n_time), track_segment_id)]
    return projected_track_positions


def calculate_replay_distance(track_graph, map_estimate, actual_positions,
                              actual_track_segment_ids, position_info,
                              center_well_id=0):
    '''Calculate the linearized distance between the replay position and the
    animal's physical position for each time point.

    Parameters
    ----------
    track_graph : networkx.Graph
        Nodes and edges describing the track
    map_estimate : ndarray, shape (n_time, n_position_dims)
        Maximum aposterior estimate of the replay
    actual_positions : ndarray, shape (n_time, 2)
        Animal's physical position during the replay
    actual_track_segment_ids : ndarray, shape (n_time,)
        Animal's track segment ID during the replay
    position_info : pandas.DataFrame
    center_well_id : hasable, optional

    Returns
    -------
    replay_distance_from_actual_position : ndarray, shape (n_time,)
    replay_distance_from_center_well : ndarray, shape (n_time,)

    '''

    actual_track_segment_ids = (
        np.asarray(actual_track_segment_ids).squeeze().astype(int))

    # Find 2D position closest to replay position
    n_position_dims = map_estimate.shape[1]
    if n_position_dims == 1:
        closest_ind = _get_closest_ind(
            map_estimate, position_info.linear_position)
    else:
        closest_ind = _get_closest_ind(
            map_estimate, position_info.loc[:, ['x_position', 'y_position']])

    df = position_info.iloc[closest_ind]
    replay_positions = df.loc[:, ['x_position', 'y_position']].values
    replay_track_segment_ids = (
        df.loc[:, ['track_segment_id']].values.squeeze().astype(int))

    track_segments = get_track_segments_from_graph(track_graph)

    # Project positions to closest edge on graph
    replay_positions = _get_projected_track_positions(
        replay_positions, track_segments, replay_track_segment_ids)
    actual_positions = _get_projected_track_positions(
        actual_positions, track_segments, actual_track_segment_ids)

    edges = np.asarray(track_graph.edges)
    replay_edge_ids = edges[replay_track_segment_ids]
    actual_edge_ids = edges[actual_track_segment_ids]
    replay_distance_from_actual_position = []
    replay_distance_from_center_well = []

    zipped = zip(
        actual_edge_ids, replay_edge_ids, actual_positions, replay_positions,
        actual_track_segment_ids, replay_track_segment_ids)

    for (actual_edge_id, replay_edge_id, actual_pos, replay_pos,
         actual_id, replay_id) in zipped:
        track_graph1 = track_graph.copy()
        if actual_id != replay_id:
            # Add actual position node
            node_name = 'actual_position'
            node1, node2 = actual_edge_id
            nx.add_path(track_graph1, [node1, node_name, node2])
            track_graph1.remove_edge(node1, node2)
            track_graph1.nodes[node_name]['pos'] = tuple(actual_pos)

            # Add replay position node
            node_name = 'replay_position'
            node1, node2 = replay_edge_id
            nx.add_path(track_graph1, [node1, node_name, node2])
            track_graph1.remove_edge(node1, node2)
            track_graph1.nodes[node_name]['pos'] = tuple(replay_pos)
        else:
            node1, node2 = actual_edge_id

            nx.add_path(track_graph1,
                        [node1, 'actual_position', 'replay_position', node2])
            nx.add_path(track_graph1,
                        [node1, 'replay_position', 'actual_position', node2])

            track_graph1.nodes['actual_position']['pos'] = tuple(actual_pos)
            track_graph1.nodes['replay_position']['pos'] = tuple(replay_pos)
            track_graph1.remove_edge(node1, node2)

        # Calculate distance between all nodes
        for edge in track_graph1.edges(data=True):
            track_graph1.edges[edge[:2]]['distance'] = np.linalg.norm(
                track_graph1.nodes[edge[0]]['pos'] -
                np.array(track_graph1.nodes[edge[1]]['pos']))

        replay_distance_from_actual_position.append(
            nx.shortest_path_length(
                track_graph1, source='actual_position',
                target='replay_position', weight='distance'))
        replay_distance_from_center_well.append(
            nx.shortest_path_length(
                track_graph1, source=center_well_id,
                target='replay_position', weight='distance'))
    replay_distance_from_actual_position = np.asarray(
        replay_distance_from_actual_position)
    replay_distance_from_center_well = np.asarray(
        replay_distance_from_center_well)

    return (replay_distance_from_actual_position,
            replay_distance_from_center_well)
