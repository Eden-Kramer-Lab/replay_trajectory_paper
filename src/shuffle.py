import numpy as np
import pandas as pd
import xarray as xr


def shuffle_run_position(position_info):
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


def shuffle_spike_time(multiunit):
    """Shuffle each tetrode spike time by a random amount
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
