import matplotlib.pyplot as plt
import numpy as np
from loren_frank_data_processing import (get_position_dataframe,
                                         get_spikes_dataframe,
                                         make_neuron_dataframe,
                                         make_tetrode_dataframe)
from loren_frank_data_processing.multiunit import (get_multiunit_dataframe,
                                                   get_multiunit_dataframe2)
from loren_frank_data_processing.position import (EDGE_ORDER, EDGE_SPACING,
                                                  make_track_graph)
from replay_trajectory_classification.core import (atleast_2d, get_track_grid,
                                                   scaled_likelihood)
from replay_trajectory_classification.multiunit_likelihood import (
    estimate_intensity, fit_occupancy, poisson_mark_log_likelihood)
from scipy.ndimage import label
from scipy.special import cotdg
from scipy.stats import multivariate_normal, rv_histogram
from skimage.transform import radon
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from src.analysis import get_map_speed
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

    neuron_info = make_neuron_dataframe(ANIMALS).xs(
        epoch_key, drop_level=False)
    neuron_info = neuron_info.loc[
        (neuron_info.numspikes > 100) &
        neuron_info.area.isin(_BRAIN_AREAS) &
        (neuron_info.type == 'principal')]

    spike_times = [np.asarray(get_spikes_dataframe(neuron_key, ANIMALS).index
                              / np.timedelta64(1, 's'))
                   for neuron_key in neuron_info.index]

    return (
        linear_position,
        multiunit_dfs,
        position_info,
        ripple_times,
        track_graph,
        center_well_id,
        spike_times,
    )


def fit_mark_likelihood(
    linear_position,
    multiunit_dfs,
    track_graph,
    center_well_id,
    model=model,
    model_kwargs=model_kwargs,
    place_bin_size=place_bin_size,
    dt=0.020
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
    n_time_bins = np.ceil((end_time - start_time) / dt).astype(int)
    time_bin_edges = start_time + np.arange(n_time_bins + 1) * dt
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


def predict_poisson_likelihood(start_time, end_time, spike_times, place_fields,
                               is_track_interior, dt=0.020):
    place_fields = np.asarray(place_fields)
    n_time_bins = np.ceil((end_time - start_time) / dt).astype(int)
    time_bin_edges = start_time + np.arange(n_time_bins + 1) * dt
    time_bin_centers = time_bin_edges[:-1] + np.diff(time_bin_edges) / 2

    spike_time_ind, neuron_ind = [], []
    for ind, times in enumerate(spike_times):
        is_valid_time = (times >= start_time) & (times <= end_time)
        inds = np.digitize(times[is_valid_time], time_bin_edges[1:-1])
        spike_time_ind.append(inds)
        neuron_ind.append(np.ones_like(inds) * ind)

    neuron_ind = np.concatenate(neuron_ind)
    spike_time_ind = np.concatenate(spike_time_ind)

    log_likelihood = np.stack(
        [np.sum(np.log(place_fields[:, neuron_ind[spike_time_ind == time_bin]] +
                       np.spacing(1)), axis=1)
         for time_bin in np.arange(n_time_bins)])
    log_likelihood -= dt * np.sum(place_fields, axis=1)

    mask = np.ones_like(is_track_interior, dtype=np.float)
    mask[~is_track_interior] = np.nan

    return scaled_likelihood(log_likelihood) * mask, time_bin_centers


def normalize_to_posterior(likelihood, prior=None):
    if prior is None:
        n_position_bins = likelihood.shape[1]
        prior = np.ones_like(likelihood) / n_position_bins
    posterior = likelihood * prior
    return posterior / np.nansum(posterior, axis=1, keepdims=True)


def convert_polar_to_slope_intercept(
    n_pixels_from_center, projection_angle, center_pixel
):
    slope = -cotdg(-projection_angle)
    intercept = (
        n_pixels_from_center / np.sin(-np.deg2rad(projection_angle))
        - slope * center_pixel[0]
        + center_pixel[1]
    )
    return intercept, slope


def detect_line_with_radon(
    posterior,
    dt,  # s
    dp,  # cm
    projection_angles=np.arange(-90, 90, 0.5),  # degrees
    filter_invalid_positions=True,
    incorporate_nearby_positions=True,
    nearby_positions_max=15,  # cm
):

    if incorporate_nearby_positions:
        n_nearby_bins = int(nearby_positions_max / 2 // dp)
        filt = np.ones(2 * n_nearby_bins + 1)
        posterior = np.apply_along_axis(
            lambda time_bin: np.convolve(time_bin, filt, mode="same"),
            axis=1, arr=posterior
        )
    else:
        n_nearby_bins = 1
    # Sinogram is shape (pixels_from_center, projection_angles)
    sinogram = radon(
        posterior.T, theta=projection_angles, circle=False,
        preserve_range=False
    )
    n_time, n_position_bins = posterior.shape
    center_pixel = np.asarray((n_time // 2, n_position_bins // 2))
    pixels_from_center = np.arange(
        -sinogram.shape[0] // 2 + 1, sinogram.shape[0] // 2 + 1)

    if filter_invalid_positions:
        start_positions, velocities = convert_polar_to_slope_intercept(
            pixels_from_center[:, np.newaxis],
            projection_angles[np.newaxis, :],
            center_pixel,
        )
        end_positions = start_positions + velocities * (n_time - 1)
        sinogram[(start_positions < 0) |
                 (start_positions > n_position_bins - 1)] = 0.0
        sinogram[(end_positions < 0) |
                 (end_positions > n_position_bins - 1)] = 0.0
        sinogram[:, np.isinf(velocities.squeeze())] = 0.0

    # Find the maximum of the sinogram
    n_pixels_from_center_ind, projection_angle_ind = np.unravel_index(
        indices=np.argmax(sinogram), shape=sinogram.shape
    )
    projection_angle = projection_angles[projection_angle_ind]
    n_pixels_from_center = pixels_from_center[n_pixels_from_center_ind]

    # Normalized score based on the integrated projection
    score = np.max(sinogram) / (n_time * n_nearby_bins)

    # Convert from polar form to slope-intercept form
    start_position, velocity = convert_polar_to_slope_intercept(
        n_pixels_from_center, projection_angle, center_pixel
    )

    # Convert from pixels to position units
    start_position *= dp
    velocity *= dp / dt

    # Estimate position for the posterior
    time = np.arange(n_time) * dt
    radon_position = start_position + velocity * time

    return start_position, velocity, radon_position, score


def map_estimate(posterior, place_bin_centers):
    posterior[np.isnan(posterior)] = 0.0
    return place_bin_centers[posterior.argmax(axis=1)].squeeze()


def _m(x, w):
    """Weighted Mean"""
    return np.sum(x * w) / np.sum(w)


def _cov(x, y, w):
    """Weighted Covariance"""
    return np.sum(w * (x - _m(x, w)) * (y - _m(y, w))) / np.sum(w)


def _corr(x, y, w):
    """Weighted Correlation"""
    return _cov(x, y, w) / np.sqrt(_cov(x, x, w) * _cov(y, y, w))


def weighted_correlation(posterior, time, place_bin_centers):
    place_bin_centers = place_bin_centers.squeeze()
    posterior[np.isnan(posterior)] = 0.0

    return _corr(time[:, np.newaxis],
                 place_bin_centers[np.newaxis, :], posterior)


def isotonic_regression(posterior, time, place_bin_centers):
    place_bin_centers = place_bin_centers.squeeze()
    posterior[np.isnan(posterior)] = 0.0

    map = map_estimate(posterior, place_bin_centers)
    map_probabilities = np.max(posterior, axis=1)

    regression = IsotonicRegression(increasing='auto').fit(
        X=time,
        y=map,
        sample_weight=map_probabilities,
    )

    score = regression.score(
        X=time,
        y=map,
        sample_weight=map_probabilities,
    )

    prediction = regression.predict(time)

    return prediction, score


def _sample_posterior(posterior, place_bin_edges, n_samples=1000):
    """Samples the posterior positions.

    Parameters
    ----------
    posterior : np.array, shape (n_time, n_position_bins)

    Returns
    -------
    posterior_samples : numpy.ndarray, shape (n_time, n_samples)

    """

    place_bin_edges = place_bin_edges.squeeze()
    n_time = posterior.shape[0]

    posterior_samples = [
        rv_histogram((posterior[time_ind], place_bin_edges)).rvs(
            size=n_samples)
        for time_ind in range(n_time)
    ]

    return np.asarray(posterior_samples)


def linear_regression(posterior, place_bin_edges, time, n_samples=1000):
    posterior[np.isnan(posterior)] = 0.0
    samples = _sample_posterior(
        posterior, place_bin_edges, n_samples=n_samples
    )
    design_matrix = np.tile(time, n_samples)[:, np.newaxis]
    response = samples.ravel(order="F")
    regression = LinearRegression().fit(X=design_matrix, y=response)

    r2 = regression.score(X=design_matrix, y=response)
    slope = regression.coef_[0]
    intercept = regression.intercept_
    prediction = regression.predict(time[:, np.newaxis])

    return intercept, slope, r2, prediction


def test_standard_decoding(
        n_time=21,
        max_position=301,
        starting_position=250.0,
        velocity=-100,
        dt=0.020,
        dp=2.0,
        use_gaussian=False):

    time = np.arange(n_time) * dt
    true_replay_position = starting_position + velocity * time

    place_bin_edges = np.arange(0, max_position + dp, dp)
    place_bin_centers = place_bin_edges[:-1] + np.diff(place_bin_edges) / 2

    likelihood = np.zeros((time.shape[0], place_bin_centers.shape[0]))
    p_ind = np.digitize(true_replay_position,
                        place_bin_edges.squeeze()[1:-1])
    if use_gaussian:
        for t_ind, peak in enumerate(place_bin_centers[p_ind]):
            likelihood[t_ind, :] = multivariate_normal(mean=peak, cov=144).pdf(
                place_bin_centers.squeeze()
            )
    else:
        t_ind = np.arange(len(time))
        likelihood[(t_ind, p_ind)] = 1.0

    posterior = normalize_to_posterior(likelihood)

    isotonic_prediction, isotonic_score = isotonic_regression(
        posterior, time, place_bin_centers
    )
    (
        start_position,
        estimated_velocity,
        radon_prediction,
        radon_score,
    ) = detect_line_with_radon(posterior, dt, dp)
    correlation = weighted_correlation(posterior, time, place_bin_centers)
    intercept, slope, linear_score, linear_prediction = linear_regression(
        posterior, place_bin_edges, time
    )

    time_bin_edges = np.append(time, time[-1] + dt)
    t, p = np.meshgrid(time_bin_edges, place_bin_edges)
    fig, axes = plt.subplots(
        2, 2, figsize=(10, 5), constrained_layout=True, sharex=True,
        sharey=True
    )
    for ax in axes.flat:
        ax.pcolormesh(t, p, likelihood.T)
        ax.scatter(time + dt / 2, true_replay_position, color="red")

    axes[0, 0].plot(time + dt / 2, map_estimate(likelihood, place_bin_centers))
    axes[0, 0].set_title(f"MAP Estimate, abs_corr={np.abs(correlation):.02f}")

    axes[0, 1].plot(time + dt / 2, isotonic_prediction)
    axes[0, 1].set_title(f"Isotonic Regression, score = {isotonic_score:.02f}")

    axes[1, 1].plot(time[:-1] + dt / 2, radon_prediction[:-1])
    axes[1, 1].set_title(f"Radon, score = {radon_score:.02f}")

    axes[1, 0].plot(time + dt / 2, linear_prediction)
    axes[1, 0].set_title(f"Linear Regression, score = {linear_score:.02f}")


def _get_max_score_metrics(metric, max_center_edge, min_left_edge,
                           min_right_edge):
    velocity1, prediction1, score1 = metric[0]
    velocity2, prediction2, score2 = metric[1]

    if score1 > score2:
        # Left Arm
        prediction1[prediction1 > max_center_edge] += (
            min_left_edge - max_center_edge
        )
        return np.abs(velocity1), prediction1, score1
    else:
        # Right Arm
        prediction2[prediction2 > max_center_edge] += (
            min_right_edge - max_center_edge
        )
        return np.abs(velocity2), prediction2, score2


def predict_clusterless_wtrack(
    time,
    likelihood,
    place_bin_centers,
    is_track_interior,
    place_bin_edges,
    track_graph1,
    place_bin_center_ind_to_node,
    dt=0.020,
):
    arm_labels = label(is_track_interior)[0]
    RIGHT_ARM = [1, 2]
    LEFT_ARM = [1, 3]
    arms = [LEFT_ARM, RIGHT_ARM]
    max_center_edge = place_bin_edges[1:][arm_labels == 1][-1][0]
    min_right_edge = place_bin_edges[:-1][arm_labels == 2][0][0]
    min_left_edge = place_bin_edges[:-1][arm_labels == 3][0][0]

    posterior = normalize_to_posterior(likelihood)
    place_bin_edges = place_bin_edges.squeeze()
    time_bin_centers = time + dt / 2
    dp = np.mean(np.diff(place_bin_edges.squeeze())[is_track_interior])

    radon = []
    isotonic = []
    linear = []
    map = []

    for arm in arms:
        arm_posterior = posterior[:, np.isin(arm_labels, arm)]

        n_position_bins = arm_posterior.shape[1]
        arm_place_bin_edges = np.arange(n_position_bins + 1) * dp
        arm_place_bin_centers = arm_place_bin_edges[:-1] + dp / 2

        (_, radon_velocity,
         radon_prediction, radon_score) = detect_line_with_radon(
            arm_posterior, dt=dt, dp=dp, incorporate_nearby_positions=False)
        radon.append((radon_velocity, radon_prediction, radon_score))

        isotonic_prediction, isotonic_score = isotonic_regression(
            arm_posterior, time_bin_centers, arm_place_bin_centers)
        isotonic_velocity = np.mean(np.diff(isotonic_prediction) / dt)
        isotonic.append(
            (isotonic_velocity, isotonic_prediction, isotonic_score))

        (_, linear_velocity, linear_score,
         linear_prediction) = linear_regression(
            arm_posterior, arm_place_bin_edges, time_bin_centers)
        linear.append((linear_velocity, linear_prediction, linear_score))

        weighted_correlation_score = weighted_correlation(
            arm_posterior, time_bin_centers, arm_place_bin_centers
        )
        map.append((0, 0, np.abs(weighted_correlation_score)))

    radon_speed, radon_prediction, radon_score = _get_max_score_metrics(
        radon, max_center_edge, min_left_edge, min_right_edge)
    (isotonic_speed, isotonic_prediction,
     isotonic_score) = _get_max_score_metrics(
        isotonic, max_center_edge, min_left_edge, min_right_edge)
    linear_speed, linear_prediction, linear_score = _get_max_score_metrics(
        linear, max_center_edge, min_left_edge, min_right_edge)
    map_score = max(map[0][-1], map[1][-1])
    map_prediction = map_estimate(posterior, place_bin_centers)
    map_speed = np.mean(get_map_speed(
        posterior,
        track_graph1,
        place_bin_center_ind_to_node,
        dt,
    ))

    return (
        time,
        likelihood,
        radon_speed,
        radon_prediction,
        radon_score,
        isotonic_speed,
        isotonic_prediction,
        isotonic_score,
        linear_speed,
        linear_prediction,
        linear_score,
        map_speed,
        map_prediction,
        map_score,
    )
