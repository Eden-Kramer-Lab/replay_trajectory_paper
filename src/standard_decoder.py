import numpy as np
from loren_frank_data_processing import (get_position_dataframe,
                                         make_tetrode_dataframe)
from loren_frank_data_processing.multiunit import (get_multiunit_dataframe,
                                                   get_multiunit_dataframe2)
from loren_frank_data_processing.position import (EDGE_ORDER, EDGE_SPACING,
                                                  make_track_graph)
from replay_trajectory_classification.core import (atleast_2d, get_track_grid,
                                                   scaled_likelihood)
from replay_trajectory_classification.multiunit_likelihood import (
    estimate_intensity, fit_occupancy, poisson_mark_log_likelihood)
from skimage.transform import radon
from src.load_data import get_ripple_times
from src.parameters import (_BRAIN_AREAS, _MARKS, ANIMALS, model, model_kwargs,
                            place_bin_size)


def load_data(epoch_key):
    position_info = get_position_dataframe(epoch_key, ANIMALS).dropna(
        subset=["linear_position", "speed"]
    )
    tetrode_info = make_tetrode_dataframe(ANIMALS, epoch_key=epoch_key)
    is_brain_areas = tetrode_info.area.astype(
        str).str.upper().isin(_BRAIN_AREAS)

    tetrode_info = tetrode_info.loc[is_brain_areas]

    multiunit_dfs = []

    for tetrode_key in tetrode_info.index:
        try:
            time = position_info.index
            multiunit_df = get_multiunit_dataframe(tetrode_key, ANIMALS).loc[
                time.min(): time.max(), _MARKS
            ]
        except AttributeError:
            multiunit_df = get_multiunit_dataframe2(
                tetrode_key, ANIMALS).loc[:, _MARKS]
        time_index = np.digitize(
            multiunit_df.index.total_seconds(),
            position_info.index.total_seconds()[1:-1],
        )
        multiunit_df["linear_position"] = np.asarray(
            position_info.iloc[time_index].linear_position)
        multiunit_df["speed"] = np.asarray(
            position_info.iloc[time_index].speed)
        multiunit_dfs.append(multiunit_df.dropna())

    is_above_speed_threshold = position_info.speed > 4
    linear_position = (position_info
                       .loc[is_above_speed_threshold]
                       .linear_position)

    track_graph, center_well_id = make_track_graph(epoch_key, ANIMALS)

    (
        ripple_times,
        _,
        _,
        _,
    ) = get_ripple_times(epoch_key)

    return (
        linear_position,
        multiunit_dfs,
        position_info,
        ripple_times,
        track_graph,
        center_well_id,
    )


def fit_mark_likelihood(
    linear_position,
    multiunit_dfs,
    track_graph,
    center_well_id,
    model=model,
    model_kwargs=model_kwargs,
    place_bin_size=place_bin_size,
):
    (
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
    ) = get_track_grid(
        track_graph, center_well_id, EDGE_ORDER, EDGE_SPACING, place_bin_size
    )

    occupancy, _ = fit_occupancy(
        linear_position, place_bin_centers, model, model_kwargs,
        is_track_interior
    )
    mean_rates = []
    ground_process_intensities = []
    joint_pdf_models = []

    dt = 0.020  # seconds
    total_time = np.diff(np.asarray(
        linear_position.index.total_seconds())[[0, -1]])
    n_total_time_bins = total_time // dt

    for multiunit_df in multiunit_dfs:
        multiunit_df = multiunit_df.loc[multiunit_df.speed > 4]

        # mean rate
        n_spikes = multiunit_df.shape[0]
        mean_rate = n_spikes / n_total_time_bins
        mean_rates.append(mean_rate)

        # ground process intensity
        marginal_pdf = np.zeros((place_bin_centers.shape[0],))

        position_at_spike = atleast_2d(
            multiunit_df.loc[multiunit_df.speed > 4].linear_position.dropna()
        )
        marginal_model = model(**model_kwargs).fit(position_at_spike)
        marginal_pdf[is_track_interior] = np.exp(
            marginal_model.score_samples(
                atleast_2d(place_bin_centers[is_track_interior])
            )
        )
        ground_process_intensity = np.zeros((1, place_bin_centers.shape[0],))
        ground_process_intensity[:, is_track_interior] = estimate_intensity(
            marginal_pdf[is_track_interior], occupancy[is_track_interior],
            mean_rate
        )
        ground_process_intensities.append(ground_process_intensity)

        # joint pdf
        marks_pos = np.asarray(
            multiunit_df.loc[
                multiunit_df.speed > 4,
                [
                    "channel_1_max",
                    "channel_2_max",
                    "channel_3_max",
                    "channel_4_max",
                    "linear_position",
                ],
            ].dropna()
        )
        joint_pdf = model(**model_kwargs).fit(marks_pos)
        joint_pdf_models.append(joint_pdf)

    return (
        occupancy,
        joint_pdf_models,
        multiunit_dfs,
        ground_process_intensities,
        mean_rates,
        is_track_interior,
        place_bin_centers,
        place_bin_edges,
        edges,
    )


def predict_mark_likelihood(
    start_time,
    end_time,
    place_bin_centers,
    occupancy,
    joint_pdf_models,
    multiunit_dfs,
    ground_process_intensities,
    mean_rates,
    is_track_interior,
    dt=0.020,
):
    time_bin_edges = np.arange(start_time, end_time + dt, dt)
    n_time_bins = len(time_bin_edges) - 1
    n_place_bins = len(place_bin_centers)

    log_likelihood = np.zeros((n_time_bins, n_place_bins))
    interior_bin_inds = np.nonzero(is_track_interior)[0]

    for joint_model, multiunit_df, gpi, mean_rate in zip(
        joint_pdf_models, multiunit_dfs, ground_process_intensities, mean_rates
    ):
        time_index = np.searchsorted(
            time_bin_edges, multiunit_df.index.total_seconds())
        in_time_bins = np.nonzero(
            ~np.isin(time_index, [0, len(time_bin_edges)]))[0]
        time_index = time_index[in_time_bins] - 1
        multiunit_df = multiunit_df.iloc[in_time_bins, :4]
        multiunit_df["time_bin_ind"] = time_index

        n_spikes = multiunit_df.shape[0]
        joint_mark_intensity = np.ones((n_spikes, n_place_bins))

        if n_spikes > 0:
            zipped = zip(
                interior_bin_inds,
                place_bin_centers[interior_bin_inds],
                occupancy[interior_bin_inds],
            )
            for bin_ind, bin, bin_occupancy in zipped:
                marks_pos = np.asarray(multiunit_df.iloc[:, :4])
                marks_pos = np.concatenate(
                    (marks_pos, bin * np.ones((n_spikes, 1))), axis=1
                )
                joint_mark_intensity[:, bin_ind] = estimate_intensity(
                    np.exp(joint_model.score_samples(marks_pos)),
                    bin_occupancy,
                    mean_rate,
                )

            tetrode_likelihood = poisson_mark_log_likelihood(
                joint_mark_intensity, np.atleast_2d(gpi)
            )
            for time_bin_ind in np.unique(time_index):
                log_likelihood[time_bin_ind] += np.sum(
                    tetrode_likelihood[time_index == time_bin_ind], axis=0
                )

    mask = np.ones_like(is_track_interior, dtype=np.float)
    mask[~is_track_interior] = np.nan

    log_likelihood = log_likelihood * mask

    time = np.arange(n_time_bins) * dt

    return scaled_likelihood(log_likelihood), time


def detect_line_with_radon(likelihood, time, place_bin_edges):
    projection_angles = np.arange(-90, 90, 0.5)  # degrees
    sinogram = radon(
        likelihood, theta=projection_angles, circle=False, preserve_range=True
    )

    center_pixel = np.asarray(
        (likelihood.shape[0] // 2, likelihood.shape[1] // 2))

    pixels_from_center = np.arange(
        -sinogram.shape[0] // 2, sinogram.shape[0] // 2)

    n_pixels_from_center_ind, projection_angle_ind = np.unravel_index(
        indices=np.nanargmax(sinogram), shape=sinogram.shape
    )

    projection_angle = projection_angles[projection_angle_ind]
    n_pixels_from_center = pixels_from_center[n_pixels_from_center_ind]
    dp = np.mean(np.diff(place_bin_edges.squeeze()))
    dt = np.mean(np.diff(time))

    projection_angle_radians = np.deg2rad(projection_angle)
    estimated_velocity = np.tan(projection_angle_radians) * dp / dt

    time_ind = np.arange(len(time))
    estimated_position = center_pixel[1] + (
        (
            n_pixels_from_center
            - (time_ind - center_pixel[0])
            * np.cos(np.pi / 2 + projection_angle_radians)
        )
        / np.sin(np.pi / 2 + projection_angle_radians)
    )
    estimated_position *= dp

    n_position_bins = likelihood.shape[1]
    score = np.nanmax(sinogram) / n_position_bins

    return estimated_velocity, estimated_position, score
