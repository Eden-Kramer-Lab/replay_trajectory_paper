import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loren_frank_data_processing import make_tetrode_dataframe
from src.figure_utilities import PAGE_HEIGHT, TWO_COLUMN, save_figure
from src.parameters import (_BRAIN_AREAS, ANIMALS, PROCESSED_DATA_DIR,
                            SHORT_STATE_ORDER, STATE_COLORS, STATE_ORDER)
from src.visualization import (SHORT_STATE_NAMES, _plot_category,
                               plot_category_duration,
                               plot_linear_position_markers,
                               plot_replay_distance_from_actual_position)
from upsetplot import UpSet


def load_replay_info():
    tetrode_info = make_tetrode_dataframe(ANIMALS)

    data_type, dim = "clusterless", "1D"
    n_unique_spiking = 2

    file_paths = glob(
        os.path.join(PROCESSED_DATA_DIR,
                     f"*_{data_type}_{dim}_replay_info.csv")
    )
    replay_info = pd.concat(
        [pd.read_csv(file_path) for file_path in file_paths], axis=0,
    ).set_index(["animal", "day", "epoch", "ripple_number"])
    replay_info = replay_info.loc[
        replay_info.n_unique_spiking >= n_unique_spiking
    ].sort_index()

    is_brain_areas = tetrode_info.area.astype(
        str).str.upper().isin(_BRAIN_AREAS)
    n_tetrodes = (
        tetrode_info.loc[is_brain_areas]
        .groupby(["animal", "day", "epoch"])
        .tetrode_id.count()
        .rename("n_tetrodes")
    )
    replay_info = pd.merge(
        replay_info.reset_index(), pd.DataFrame(n_tetrodes).reset_index()
    ).set_index(["animal", "day", "epoch", "ripple_number"])

    for state in STATE_ORDER:
        replay_info[f"{state}_pct_unique_spiking"] = (
            replay_info[f"{state}_n_unique_spiking"] /
            replay_info["n_tetrodes"]
        )
    replay_info = replay_info.rename(index={"Cor": "cor"}).rename_axis(
        index={"animal": "Animal ID"}
    )

    return replay_info


def plot_category_counts(replay_info):
    df = (replay_info
          .loc[replay_info.is_classified]
          .rename(columns=SHORT_STATE_NAMES)
          .set_index(SHORT_STATE_ORDER[::-1]))
    upset = UpSet(
        df,
        sort_sets_by=None,
        show_counts=False,
        subset_size="count",
        sort_by="cardinality",
        intersection_plot_elements=5,
    )
    ax_dict = upset.plot()
    n_classified = replay_info.is_classified.sum()
    _, intersect_max = ax_dict["intersections"].get_ylim()
    ax_dict["intersections"].set_yticks(n_classified * np.arange(0, 0.6, 0.1))
    ax_dict["intersections"].set_yticklabels(range(0, 60, 10))
    ax_dict["intersections"].set_ylabel(
        "Percentage\nof Ripples",
        ha="center",
        va="center",
        rotation="horizontal",
        labelpad=30,
    )
    ax_dict["intersections"].text(
        9, n_classified * 0.45, f"N = {n_classified}", zorder=1000, fontsize=9
    )

    ax_dict["totals"].set_xticks([0, 0.5 * n_classified])
    ax_dict["totals"].set_xticklabels([0, 50])
    ax_dict["totals"].set_xlabel("Marginal Percentage\nof Ripples")
    ax_dict["totals"].set_ylim([-0.5, 4.4])

    plt.suptitle("Most Common Combinations of Dynamics",
                 fontsize=14, x=0.55, y=0.925)
    for i, color in enumerate(STATE_ORDER):
        rect = plt.Rectangle(
            xy=(0, len(STATE_ORDER) - i - 1.4),
            width=1,
            height=0.8,
            facecolor=STATE_COLORS[color],
            lw=0,
            zorder=0,
            alpha=0.25,
        )
        ax_dict["shading"].add_patch(rect)

    save_figure(os.path.join("Figure4", "figure4_category_counts"))


def plot_stats(replay_info, saturation=0.7, fliersize=1.0):

    fig, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(TWO_COLUMN, PAGE_HEIGHT / 2),
        constrained_layout=True
    )

    # Duration of Dynamic
    plot_category_duration(
        replay_info, kind="box", ax=axes[0, 0], fliersize=fliersize,
        saturation=saturation,
    )
    axes[0, 0].set_title("Duration of Dynamic")

    # Distance from Animal
    plot_replay_distance_from_actual_position(
        replay_info, kind="box", ax=axes[0, 1], fliersize=fliersize,
        saturation=saturation
    )
    axes[0, 1].set_title("Distance from Animal")
    axes[0, 1].set_yticklabels([])

    # Non-Local Hover Position
    norm_non_local_hover = (
        replay_info.loc[
            replay_info.Hover_replay_distance_from_actual_position > 30
        ].Hover_replay_linear_position.values
        / replay_info.loc[
            replay_info.Hover_replay_distance_from_actual_position > 30
        ].left_well_position
    )

    sns.distplot(
        norm_non_local_hover,
        kde_kws=dict(
            bw=0.025, cut=0, shade=True, facecolor=STATE_COLORS["Hover"],
            legend=False,
        ),
        rug_kws=dict(color="black", alpha=0.5),
        kde=True,
        rug=True,
        hist=False,
        color=STATE_COLORS["Hover"],
        ax=axes[1, 0],
    )
    axes[1, 0].set_xlabel("Normalized Position")
    axes[1, 0].set_ylabel("Probability Density")
    plot_linear_position_markers(
        replay_info,
        is_normalized=True,
        jitter=0.00,
        zorder=101,
        alpha=1,
        ax=axes[1, 0],
        linestyle="-",
        fontsize=14,
    )

    sns.despine(ax=axes[1, 0])
    axes[1, 0].set_xlim((0, 1))
    axes[1, 0].set_title("Non-Local Hover Position")
    n_non_local = norm_non_local_hover.size
    axes[1, 0].text(0.75, 3.5, f"N = {n_non_local}", zorder=100, fontsize=6)

    # Population firing rate
    _plot_category(
        replay_info,
        "population_rate",
        kind="box",
        ax=axes[1, 1],
        fliersize=fliersize,
        saturation=saturation,
    )
    axes[1, 1].set_yticklabels([])
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xlim((1e0, 1e6))
    axes[1, 1].set_xlabel("Rate [spikes / s]")
    axes[1, 1].set_title("Multiunit Population Rate")

    save_figure(os.path.join("Figure4", "figure4_dynamics_summary"))
