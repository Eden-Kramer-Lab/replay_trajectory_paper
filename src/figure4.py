import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from src.analysis import (get_is_classified, get_probability,
                          highest_posterior_density)
from src.figure_utilities import TWO_COLUMN, PAGE_HEIGHT, save_figure
from src.parameters import PROBABILITY_THRESHOLD, STATE_COLORS
from src.visualization import plot_1D_wtrack_landmarks

MILLISECONDS_TO_SECONDS = 1000


def plot_clusterless_1D_results_hpd(
    data,
    results,
    classifier,
    epoch_key,
    ripple_number,
    cmap="bone_r",
    is_save_figure=True,
):
    time_slice = slice(
        *data["ripple_times"].loc[ripple_number, ["start_time", "end_time"]]
    )

    ripple_duration = (
        MILLISECONDS_TO_SECONDS
        * (time_slice.stop - time_slice.start)
        / np.timedelta64(1, "s")
    )
    
    max_position = np.ceil(data["position_info"].linear_position.max()).astype(int)

    fig, axes = plt.subplots(
        3,
        1,
        constrained_layout=True,
        figsize=(TWO_COLUMN / 4, 0.8 * PAGE_HEIGHT / 3),
        gridspec_kw={"height_ratios": [1, 3, 1]},
    )

    # axis 0
    probability = results.acausal_posterior.sum(["position"])
    time = MILLISECONDS_TO_SECONDS * probability.time / np.timedelta64(1, "s")
    max_time = time.max()

    for state, prob in zip(results.state.values, probability.values.T):
        axes[0].plot(
            time, prob, linewidth=1, color=STATE_COLORS[state], clip_on=False
        )

    axes[0].set_ylim((0, 1))
    axes[0].set_yticks((0, 1))
    axes[0].set_ylabel("Prob.")
    axes[0].set_xlim((0, max_time))
    axes[0].set_xticks([])
    sns.despine(ax=axes[0], offset=5)
    axes[0].spines["bottom"].set_visible(False)
    
    probability2 = get_probability(results)
    is_classified = get_is_classified(probability2, PROBABILITY_THRESHOLD)

    for state, is_class in zip(is_classified.state.values, is_classified.values.T):
        if is_class.sum() > 0:
            axes[0].fill_between(
                time,
                is_class,
                where=is_class.astype(bool),
                alpha=0.25,
                color=STATE_COLORS[state],
            )
    axes[0].set_xlabel("")

    # axis 1
    posterior = results.assign_coords(
        time=time
    ).acausal_posterior.sum("state")
    cmap = copy.copy(plt.cm.get_cmap(cmap))
    cmap.set_bad(color="lightgrey", alpha=1.0)
    (
        posterior.where(classifier.is_track_interior_).plot(
            x="time",
            y="position",
            robust=True,
            add_colorbar=False,
            zorder=0,
            rasterized=True,
            cmap=cmap,
            ax=axes[1],
        )
    )
    axes[1].set_title("")

    ripple_position = data["position_info"].loc[time_slice, "linear_position"]
    axes[1].plot(time, ripple_position, linestyle="--", linewidth=2,
                 color="magenta", clip_on=False)
    axes[1].set_xlim((0, max_time))
    axes[1].set_xticks([])
    axes[1].set_ylabel("Pos. [cm]")
    axes[1].set_xlabel("")
    axes[1].set_ylim((0, max_position))
    axes[1].set_yticks((0, max_position))
    sns.despine(ax=axes[1], offset=5)
    axes[1].spines["bottom"].set_visible(False)
    
    # axis 2
    hpd_threshold = highest_posterior_density(posterior, coverage=0.95)
    isin_hpd = posterior >= hpd_threshold[:, np.newaxis]
    spatial_coverage = (
        (isin_hpd * np.diff(posterior.position)[0]).sum("position").values
    )
    axes[2].plot(time, spatial_coverage, color="grey", clip_on=False, linewidth=1)
    axes[2].fill_between(time, spatial_coverage, color="lightgrey", clip_on=False, alpha=1)
    axes[2].set_ylabel("95% HPD\n[cm]")
    axes[2].set_xlim((0, max_time))
    axes[2].set_xlabel("Time [ms]")
    axes[2].set_xticks((0, np.round(ripple_duration).astype(int)))
    axes[2].set_ylim((0, max_position))
    axes[2].set_yticks((0, max_position))
    sns.despine(ax=axes[2], offset=5)

    # Save Plot
    if is_save_figure:
        animal, day, epoch = epoch_key
        fig_name = (
            "figure4_"
            f"{animal}_{day:02d}_{epoch:02d}_{ripple_number:04d}_"
            f"clusterless_1D_spatial_coverage"
        )
        save_figure(os.path.join("Figure4", fig_name))
