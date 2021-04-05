import numpy as np
import pandas as pd
import xarray as xr
from replay_trajectory_classification.multiunit_likelihood import (
    estimate_ground_process_intensity, train_joint_model)


def shuffle_likelihood_position_bins(likelihood, is_track_interior):
    n_position_bins = likelihood.shape[1]
    midpoint = (n_position_bins - 1) // 2
    shuffled_likelihood = np.zeros_like(likelihood)

    shuffled_likelihood[:, is_track_interior] = np.stack(
        [
            np.roll(time_bin, shift=np.random.randint(
                low=-1 * midpoint, high=midpoint))
            for time_bin in likelihood[:, is_track_interior]
        ],
        axis=0,
    )
    return shuffled_likelihood


def shuffle_run_position(position_info):
    """Circularly shuffle position within each run (from well to well)"""
    n_time = position_info.shape[0]
    shuffled_position = pd.Series(
        np.full((n_time,), np.nan), index=position_info.index)

    for segment, df in position_info.groupby("labeled_segments"):
        n_segment_time = df.shape[0]
        if (segment == 0) | (n_segment_time == 1):
            shift = 0
        elif n_segment_time == 2:
            shift = np.random.randint(low=0, high=2)
        else:
            midpoint = (n_segment_time - 1) // 2
            shift = np.random.randint(low=-1 * midpoint, high=midpoint)

        shuffled_position.loc[df.index] = np.roll(
            df.linear_position, shift=shift)

    return shuffled_position


def jitter_spike_time(multiunit):
    """Jitter each tetrode spike time by a random amount
    """
    n_time = len(multiunit.time)
    n_tetrodes = len(multiunit.tetrodes)

    rand_time_offset = np.random.randint(
        low=-(n_time - 1) // 2, high=(n_time - 1) // 2, size=n_tetrodes
    )
    shuffled_multiunit = [
        multiunit.isel(tetrodes=tetrode_ind, drop=False).roll(
            time=time_offset_ind, roll_coords=False
        )
        for tetrode_ind, time_offset_ind in enumerate(rand_time_offset)
    ]

    return xr.concat(shuffled_multiunit, dim=multiunit.tetrodes).transpose(
        *multiunit.dims
    )


def tetrode_identity_shuffle(multiunit):
    """Shuffle the correspondence between tetrodes and spike time
    """
    n_tetrodes = len(multiunit.tetrodes)
    rand_tetrode_index = np.random.randint(
        low=0, high=n_tetrodes, size=n_tetrodes)
    return multiunit.isel(tetrodes=rand_tetrode_index)


def shuffle_segments_run_position(position_info, multiunits, classifier):
    is_training = (position_info.speed > 4) & (
        position_info.labeled_segments != 0)
    position_info = position_info.loc[is_training]
    segments = position_info.labeled_segments.unique()
    multiunits = multiunits.values[is_training]
    joint_pdf_models = []
    ground_process_intensities = []

    for tetrode_ind, multiunit in enumerate(np.moveaxis(multiunits, -1, 0)):
        shuffled_position = shuffle_run_position(position_info)
        shuffled_position = np.concatenate(
            [np.asarray(
                shuffled_position[position_info.labeled_segments == segment])
             for segment in np.random.choice(segments, size=segments.size,
                                             replace=False)])
        ground_process_intensities.append(
            estimate_ground_process_intensity(
                multiunit, shuffled_position[:, np.newaxis],
                classifier.place_bin_centers_, classifier.occupancy_,
                classifier.mean_rates_[tetrode_ind], classifier.model,
                classifier.model_kwargs, classifier.is_track_interior_))
        joint_pdf_models.append(
            train_joint_model(multiunit, shuffled_position[:, np.newaxis],
                              classifier.model, classifier.model_kwargs))

    shuffle_classifier = classifier.copy()
    shuffle_classifier.ground_process_intensities_ = ground_process_intensities
    shuffle_classifier.joint_pdf_models_ = joint_pdf_models

    return shuffle_classifier


def get_shuffled_pvalue(observed_stat, shuffled_stat, type='two-sided'):
    if type == 'two-sided':
        n_greater = np.sum(np.abs(shuffled_stat) >= np.abs(observed_stat))
    elif type == 'one-sided-greater':
        n_greater = np.sum(shuffled_stat >= observed_stat)
    elif type == 'one-sided-lesser':
        n_greater = np.sum(shuffled_stat <= observed_stat)

    n_shuffles = len(shuffled_stat)
    return (n_greater + 1) / (n_shuffles + 1)


def Benjamini_Hochberg_procedure(p_values, alpha=0.05):
    '''Corrects for multiple comparisons and returns the significant
    p-values by controlling the false discovery rate at level `alpha`
    using the Benjamani-Hochberg procedure.
    Parameters
    ----------
    p_values : array_like
    alpha : float, optional
        The expected proportion of false positive tests.
    Returns
    -------
    is_significant : boolean nd-array
        A boolean array the same shape as `p_values` indicating whether the
        null hypothesis has been rejected (True) or failed to reject
        (False).
    '''
    p_values = np.array(p_values)
    threshold_line = np.linspace(0, alpha, num=p_values.size + 1,
                                 endpoint=True)[1:]
    sorted_p_values = np.sort(p_values.flatten())
    try:
        threshold_ind = np.max(
            np.where(sorted_p_values <= threshold_line)[0])
        threshold = sorted_p_values[threshold_ind]
    except ValueError:  # There are no values below threshold
        threshold = -1
    return p_values <= threshold


def Bonferroni_correction(p_values, alpha=0.05):
    p_values = np.asarray(p_values)
    return p_values <= alpha / p_values.size
