import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from src.analysis import get_is_classified, get_probability
from src.figure_utilities import ONE_COLUMN, PAGE_HEIGHT, save_figure
from src.parameters import PROBABILITY_THRESHOLD, STATE_COLORS
from src.visualization import (plot_1D_wtrack_landmarks,
                               plot_2D_position_with_color_time)

MILLISECONDS_TO_SECONDS = 1000


def plot_1D_results(spike_times, data, results_1D,
                    classifier_1D, epoch_key,
                    ripple_number, data_type="sorted_spikes", cmap="viridis"):
    fig, axes = plt.subplots(
        3,
        1,
        sharex=True,
        constrained_layout=True,
        figsize=(0.9 * ONE_COLUMN, 0.9 * PAGE_HEIGHT / 3),
        gridspec_kw={"height_ratios": [1, 1, 3]},
    )

    n_tetrodes = len(spike_times)

    animal, day, epoch = epoch_key

    ripple_start, ripple_end = (
        data["ripple_times"].loc[ripple_number].start_time,
        data["ripple_times"].loc[ripple_number].end_time,
    )

    ripple_position_info = data["position_info"].loc[ripple_start:ripple_end]

    # axis 0
    axes[0].eventplot(
        [
            MILLISECONDS_TO_SECONDS
            * (multiunit.loc[ripple_start:ripple_end].index - ripple_start)
            / np.timedelta64(1, "s")
            for multiunit in spike_times
        ],
        color="black",
    )

    axes[0].set_xlim(
        (0, MILLISECONDS_TO_SECONDS *
         (ripple_end - ripple_start) / np.timedelta64(1, "s"))
    )
    axes[0].set_xticks(
        (0, MILLISECONDS_TO_SECONDS *
         (ripple_end - ripple_start) / np.timedelta64(1, "s"))
    )

    axes[0].set_yticks((0, n_tetrodes))

    if data_type == "sorted_spikes":
        axes[0].set_ylabel("Cells")
    else:
        axes[0].set_ylabel("Tet.")

    # axis 1
    probability = results_1D.acausal_posterior.sum(["position"])

    for state, prob in zip(results_1D.state.values,
                           probability.values.T):
        axes[1].plot(
            MILLISECONDS_TO_SECONDS *
            probability.time / np.timedelta64(1, "s"),
            prob,
            linewidth=2,
            color=STATE_COLORS[state],
        )

    axes[1].set_ylim((0, 1.05))
    axes[1].set_yticks((0, 1))
    axes[1].set_ylabel("Prob.")

    probability2 = get_probability(results_1D)
    is_classified = get_is_classified(probability2, PROBABILITY_THRESHOLD)

    time = MILLISECONDS_TO_SECONDS * probability.time / np.timedelta64(1, "s")

    for state, is_class in zip(is_classified.state.values,
                               is_classified.values.T):
        if is_class.sum() > 0:
            axes[1].fill_between(
                time,
                is_class,
                where=is_class.astype(bool),
                alpha=0.25,
                color=STATE_COLORS[state],
            )

    # axis 2

    cmap = copy.copy(plt.cm.get_cmap(cmap))
    cmap.set_bad(color="lightgrey", alpha=1.0)

    (
        results_1D.assign_coords(
            time=lambda ds: MILLISECONDS_TO_SECONDS *
            ds.time / np.timedelta64(1, "s")
        )
        .acausal_posterior.sum("state")
        .where(classifier_1D.is_track_interior_)
        .plot(
            x="time",
            y="position",
            robust=True,
            add_colorbar=False,
            zorder=0,
            rasterized=True,
            cmap=cmap,
        )
    )

    ripple_position = ripple_position_info.loc[:, "linear_position"]
    max_time = (MILLISECONDS_TO_SECONDS * probability.time /
                np.timedelta64(1, "s")).max()
    axes[2].plot(time, ripple_position, linestyle="--", linewidth=2,
                 color="magenta", clip_on=False)
    axes[2].set_xlim((0, max_time))
    axes[2].set_xticks(
        (0, MILLISECONDS_TO_SECONDS *
         (ripple_end - ripple_start) / np.timedelta64(1, "s"))
    )
    axes[2].set_xlabel("Time [ms]")

    plot_1D_wtrack_landmarks(data, max_time, ax=axes[2])
    axes[2].set_ylabel("Position [cm]")

    sns.despine(offset=5)

    fig_name = (
        "figure2_"
        f"{animal}_{day:02d}_{epoch:02d}_{ripple_number:04d}_"
        f"{data_type}_1D_acasual_classification"
    )
    save_figure(os.path.join("Figure2", fig_name))


def plot_1D_projected_to_2D(epoch_key, ripple_number, data,
                            results, classifier, data_type="clusterless",
                            posterior_time_cmap="cool"):
    animal, day, epoch = epoch_key

    position_2D = data["position_info"].loc[:, ["x_position", "y_position"]]

    ripple_start, ripple_end = (
        data["ripple_times"].loc[ripple_number].start_time,
        data["ripple_times"].loc[ripple_number].end_time,
    )

    ripple_position_info = data["position_info"].loc[ripple_start:ripple_end]

    map_position_ind = (
        results.sum(
            "state").acausal_posterior.argmax("position").values
    )
    map_position_2d = classifier.place_bin_center_2D_position_[
        map_position_ind
    ]

    fig, ax = plt.subplots(1, 1, figsize=(
        ONE_COLUMN * 0.75, ONE_COLUMN * 0.75), constrained_layout=True)
    ax.plot(
        position_2D.values[:, 0],
        position_2D.values[:, 1],
        color="lightgrey",
        alpha=0.4,
        zorder=1,
    )

    _, _, cbar = plot_2D_position_with_color_time(
        MILLISECONDS_TO_SECONDS *
        results.time / np.timedelta64(1, "s"),
        map_position_2d,
        ax=ax,
        cmap=posterior_time_cmap,
    )
    cbar.set_label("Time [ms]")
    cbar.outline.set_visible(False)

    ax.scatter(
        ripple_position_info["projected_x_position"],
        ripple_position_info["projected_y_position"],
        zorder=100,
        color="magenta",
        s=100,
    )
    ax.text(183, 158, "L", ha="center", va="center", weight="bold", zorder=100)
    ax.text(218, 158, "C", ha="center", va="center", weight="bold", zorder=100)
    ax.text(252, 158, "R", ha="center", va="center", weight="bold", zorder=100)

    ax.set_xlim((position_2D.values[:, 0].min(),
                 position_2D.values[:, 0].max()))
    ax.set_xticks(
        (np.ceil(position_2D.values[:, 0].min()),
         np.ceil(position_2D.values[:, 0].max()))
    )
    ax.set_xlabel("X-Position [cm]")

    ax.set_ylim((position_2D.values[:, 1].min(),
                 position_2D.values[:, 1].max()))
    ax.set_yticks(
        (np.ceil(position_2D.values[:, 1].min()),
         np.ceil(position_2D.values[:, 1].max()))
    )
    ax.set_ylabel("Y-Position [cm]")

    ax.axis("square")
    sns.despine()

    fig_name = (
        "figure2_1D_projected_to_2D_"
        f"{animal}_{day:02d}_{epoch:02d}_{ripple_number:04d}_"
        f"{data_type}_1D_acasual_classification"
    )
    save_figure(os.path.join("Figure2", fig_name))
