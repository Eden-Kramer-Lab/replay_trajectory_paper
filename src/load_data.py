import itertools
from logging import getLogger

import numpy as np
import pandas as pd
from loren_frank_data_processing import (get_all_multiunit_indicators,
                                         get_all_spike_indicators,
                                         get_interpolated_position_dataframe,
                                         get_LFPs, get_position_dataframe,
                                         get_trial_time, make_epochs_dataframe,
                                         make_neuron_dataframe,
                                         make_tetrode_dataframe)
from ripple_detection import (Kay_ripple_detector,
                              get_multiunit_population_firing_rate)
from ripple_detection.core import _get_ripplefilter_kernel
from scipy.signal import filtfilt

from .parameters import _BRAIN_AREAS, _MARKS, ANIMALS, SAMPLING_FREQUENCY

logger = getLogger(__name__)


def filter_ripple_band(data, sampling_frequency=1500):
    '''Returns a bandpass filtered signal between 150-250 Hz

    Parameters
    ----------
    data : array_like, shape (n_time,)

    Returns
    -------
    filtered_data : array_like, shape (n_time,)

    '''
    filter_numerator, filter_denominator = _get_ripplefilter_kernel()
    is_nan = np.isnan(data)
    filtered_data = np.full_like(data, np.nan)
    filtered_data[~is_nan] = filtfilt(
        filter_numerator, filter_denominator, data[~is_nan], axis=0)
    return filtered_data


def get_ripple_times(epoch_key, sampling_frequency=1500,
                     brain_areas=['CA1', 'CA2', 'CA3']):
    position_info = (
        get_interpolated_position_dataframe(epoch_key, ANIMALS)
        .dropna(subset=['linear_position', 'speed']))
    speed = position_info['speed']
    time = position_info.index
    tetrode_info = make_tetrode_dataframe(ANIMALS).xs(
        epoch_key, drop_level=False)
    if ~np.all(np.isnan(tetrode_info.validripple.astype(float))):
        tetrode_keys = tetrode_info.loc[
            (tetrode_info.validripple == 1)].index
    else:
        is_brain_areas = (
            tetrode_info.area.astype(str).str.upper().isin(brain_areas))
        tetrode_keys = tetrode_info.loc[is_brain_areas].index

    lfps = get_LFPs(tetrode_keys, ANIMALS).reindex(time)
    ripple_filtered_lfps = pd.DataFrame(
        np.stack([filter_ripple_band(
            lfps.values[:, ind], sampling_frequency=1500)
            for ind in np.arange(lfps.shape[1])], axis=1),
        index=lfps.index)

    ripple_times = Kay_ripple_detector(
        time, lfps.values, speed.values, sampling_frequency,
        zscore_threshold=2.0, close_ripple_threshold=np.timedelta64(0, 'ms'),
        minimum_duration=np.timedelta64(15, 'ms'))

    return ripple_times, ripple_filtered_lfps, lfps


def load_data(epoch_key, brain_areas=None):

    if brain_areas is None:
        brain_areas = _BRAIN_AREAS

    time = get_trial_time(epoch_key, ANIMALS)
    time = (pd.Series(np.ones_like(time, dtype=np.float), index=time)
            .resample('2ms').mean()
            .index)

    def _time_function(*args, **kwargs):
        return time

    position_info = (
        get_interpolated_position_dataframe(
            epoch_key, ANIMALS, _time_function)
        .dropna(subset=['linear_position', 'speed']))

    time = position_info.index

    tetrode_info = make_tetrode_dataframe(ANIMALS, epoch_key=epoch_key)
    is_brain_areas = (
        tetrode_info.area.astype(str).str.upper().isin(brain_areas))
    tetrode_keys = tetrode_info.loc[is_brain_areas].index
    lfps = get_LFPs(tetrode_keys, ANIMALS)
    lfps = lfps.resample('2ms').mean().fillna(method='pad').reindex(time)

    try:
        neuron_info = make_neuron_dataframe(ANIMALS).xs(
            epoch_key, drop_level=False)
        neuron_info = neuron_info.loc[
            (neuron_info.numspikes > 100) &
            neuron_info.area.isin(brain_areas) &
            (neuron_info.type == 'principal')]
        spikes = get_all_spike_indicators(
            neuron_info.index, ANIMALS, _time_function).reindex(time)
    except KeyError:
        spikes = None

    tetrode_info = tetrode_info.loc[is_brain_areas]
    multiunit = (get_all_multiunit_indicators(
        tetrode_info.index, ANIMALS, _time_function)
        .sel(features=_MARKS)
        .reindex({'time': time}))
    multiunit_spikes = (np.any(~np.isnan(multiunit.values), axis=1)
                        ).astype(np.float)
    multiunit_firing_rate = pd.DataFrame(
        get_multiunit_population_firing_rate(
            multiunit_spikes, SAMPLING_FREQUENCY), index=time,
        columns=['firing_rate'])

    logger.info('Finding ripple times...')
    ripple_times, ripple_filtered_lfps, ripple_lfps = get_ripple_times(
        epoch_key)

    ripple_times = ripple_times.assign(
        duration=lambda df: (df.end_time - df.start_time).dt.total_seconds())

    return {
        'position_info': position_info,
        'ripple_times': ripple_times,
        'spikes': spikes,
        'multiunit': multiunit,
        'lfps': lfps,
        'tetrode_info': tetrode_info,
        'ripple_filtered_lfps': ripple_filtered_lfps,
        'ripple_lfps': ripple_lfps,
        'multiunit_firing_rate': multiunit_firing_rate,
        'sampling_frequency': SAMPLING_FREQUENCY,
    }


def get_sleep_interpolated_position_info(
    epoch_key, animals, time_function=get_trial_time
):
    time = time_function(epoch_key, ANIMALS)
    position_df = get_position_dataframe(
        epoch_key, animals, skip_linearization=True)
    new_index = pd.Index(
        np.unique(np.concatenate((position_df.index, time))), name="time"
    )
    position_df = (
        position_df.reindex(index=new_index)
        .interpolate(method="linear")
        .reindex(index=time)
    )

    position_df.loc[position_df.speed < 0, "speed"] = 0.0

    return position_df


def get_sleep_ripple_times(epoch_key, brain_areas=["CA1", "CA2", "CA3"]):
    position_info = (get_sleep_interpolated_position_info(epoch_key, ANIMALS)
                     .dropna(subset=["speed"]))
    speed = position_info["speed"]
    time = position_info.index
    tetrode_info = make_tetrode_dataframe(
        ANIMALS).xs(epoch_key, drop_level=False)
    if ~np.all(np.isnan(tetrode_info.validripple.astype(float))):
        tetrode_keys = tetrode_info.loc[(tetrode_info.validripple == 1)].index
    else:
        is_brain_areas = tetrode_info.area.astype(
            str).str.upper().isin(brain_areas)
        tetrode_keys = tetrode_info.loc[is_brain_areas].index

    lfps = get_LFPs(tetrode_keys, ANIMALS).reindex(time)
    ripple_filtered_lfps = pd.DataFrame(
        np.stack(
            [
                filter_ripple_band(
                    lfps.values[:, ind], sampling_frequency=1500)
                for ind in np.arange(lfps.shape[1])
            ],
            axis=1,
        ),
        index=lfps.index,
    )

    ripple_times = Kay_ripple_detector(
        time,
        lfps.values,
        speed.values,
        sampling_frequency=1500,
        zscore_threshold=2.0,
        close_ripple_threshold=np.timedelta64(0, "ms"),
        minimum_duration=np.timedelta64(15, "ms"),
    )

    return ripple_times, ripple_filtered_lfps, lfps


def load_sleep_data(epoch_key, brain_areas=None):

    if brain_areas is None:
        brain_areas = _BRAIN_AREAS

    time = get_trial_time(epoch_key, ANIMALS)
    time = (
        pd.Series(np.ones_like(time, dtype=np.float), index=time)
        .resample("2ms")
        .mean()
        .index
    )

    def _time_function(*args, **kwargs):
        return time

    position_info = get_sleep_interpolated_position_info(
        epoch_key, ANIMALS, _time_function
    ).dropna(subset=["speed"])

    time = position_info.index

    tetrode_info = make_tetrode_dataframe(ANIMALS, epoch_key=epoch_key)
    is_brain_areas = tetrode_info.area.astype(
        str).str.upper().isin(brain_areas)
    tetrode_keys = tetrode_info.loc[is_brain_areas].index
    lfps = get_LFPs(tetrode_keys, ANIMALS)
    lfps = lfps.resample("2ms").mean().fillna(method="pad").reindex(time)

    try:
        neuron_info = make_neuron_dataframe(
            ANIMALS).xs(epoch_key, drop_level=False)
        neuron_info = neuron_info.loc[
            (neuron_info.numspikes > 100)
            & neuron_info.area.isin(brain_areas)
            & (neuron_info.type == "principal")
        ]
        spikes = get_all_spike_indicators(
            neuron_info.index, ANIMALS, _time_function
        ).reindex(time)
    except KeyError:
        spikes = None

    tetrode_info = tetrode_info.loc[is_brain_areas]
    multiunit = (
        get_all_multiunit_indicators(
            tetrode_info.index, ANIMALS, _time_function)
        .sel(features=_MARKS)
        .reindex({"time": time})
    )
    multiunit_spikes = (
        np.any(~np.isnan(multiunit.values), axis=1)).astype(np.float)
    multiunit_firing_rate = pd.DataFrame(
        get_multiunit_population_firing_rate(
            multiunit_spikes, SAMPLING_FREQUENCY),
        index=time,
        columns=["firing_rate"],
    )

    ripple_times, ripple_filtered_lfps, ripple_lfps = get_sleep_ripple_times(
        epoch_key)

    ripple_times = ripple_times.assign(
        duration=lambda df: (df.end_time - df.start_time).dt.total_seconds()
    )

    return {
        "position_info": position_info,
        "ripple_times": ripple_times,
        "spikes": spikes,
        "multiunit": multiunit,
        "lfps": lfps,
        "tetrode_info": tetrode_info,
        "ripple_filtered_lfps": ripple_filtered_lfps,
        "ripple_lfps": ripple_lfps,
        "multiunit_firing_rate": multiunit_firing_rate,
        "sampling_frequency": SAMPLING_FREQUENCY,
    }


def get_sleep_and_prev_run_epochs():

    epoch_info = make_epochs_dataframe(ANIMALS)
    sleep_epoch_keys = []
    prev_run_epoch_keys = []

    for _, df in epoch_info.groupby(["animal", "day"]):
        is_w_track = (
            df.iloc[:-1].environment.isin(
                ["TrackA", "TrackB", "WTrackA", "WTrackB", "wtrack"]
            )
        ).values

        is_sleep_after_run = (df.iloc[1:].type == "sleep") & is_w_track
        sleep_ind = np.nonzero(is_sleep_after_run)[0] + 1

        sleep_epoch_keys.append(df.iloc[sleep_ind].index)
        prev_run_epoch_keys.append(df.iloc[sleep_ind - 1].index)

    sleep_epoch_keys = list(itertools.chain(*sleep_epoch_keys))
    prev_run_epoch_keys = list(itertools.chain(*prev_run_epoch_keys))

    return sleep_epoch_keys, prev_run_epoch_keys
