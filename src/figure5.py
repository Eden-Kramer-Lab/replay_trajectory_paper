import ast
import os
import re
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loren_frank_data_processing import make_tetrode_dataframe
from src.figure_utilities import PAGE_HEIGHT, TWO_COLUMN, save_figure
from src.parameters import (_BRAIN_AREAS, ANIMALS, PROBABILITY_THRESHOLD,
                            PROCESSED_DATA_DIR, SHORT_STATE_ORDER,
                            STATE_COLORS, STATE_ORDER)
from src.visualization import (SHORT_STATE_NAMES, _plot_category,
                               plot_category_duration,
                               plot_linear_position_markers,
                               plot_replay_distance_from_actual_position)
from upsetplot import UpSet


def load_replay_info(
    n_unique_spiking=2,
    data_type="clusterless",
    dim="1D",
    probability_threshold=PROBABILITY_THRESHOLD,
    speed_threshold=4,
    exclude_interneuron_spikes=False
):
    tetrode_info = make_tetrode_dataframe(ANIMALS)
    prob = int(probability_threshold * 100)
    if exclude_interneuron_spikes:
        interneuron = 'no_interneuron_'
    else:
        interneuron = ''
    file_regex = f"*_{data_type}_{dim}_{interneuron}replay_info_{prob:02d}.csv"
    file_paths = glob(os.path.join(PROCESSED_DATA_DIR, file_regex))
    replay_info = pd.concat(
        [pd.read_csv(file_path) for file_path in file_paths], axis=0,
    ).set_index(["animal", "day", "epoch", "ripple_number"])
    replay_info = replay_info.loc[
        (replay_info.n_unique_spiking >= n_unique_spiking) &
        (replay_info.actual_speed <= speed_threshold)
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
    ax_dict["totals"].set_ylim([-0.5, len(SHORT_STATE_ORDER) - 1 + 0.4])

    plt.suptitle("Most Common Combinations of Classifications per Ripple",
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

    save_figure(os.path.join("Figure5", "figure5_category_counts"))


def convert_object_to_array(fixed_string):
    pattern = r"""# Match (mandatory) whitespace between...
                  (?<=\]) # ] and
                  \s+
                  (?= \[) # [, or
                  |
                  (?<=[^\[\]\s])
                  \s+
                  (?= [^\[\]\s]) # two non-bracket non-whitespace characters
               """

    # Replace such whitespace with a comma
    fixed_string = re.sub(pattern, ",", fixed_string, flags=re.VERBOSE)

    return np.array(ast.literal_eval(fixed_string))


def get_norm_linear_position(replay_info):
    non_local_stationary = replay_info.loc[
        replay_info.Hover_replay_distance_from_actual_position > 30
    ]
    norm_linear_position = []
    for ripple_id, df in non_local_stationary.iterrows():
        try:
            temp = (
                convert_object_to_array(df.Hover_replay_linear_position)
                / df.left_well_position
            )
            for pos in temp:
                norm_linear_position.append(pos)
        except TypeError:
            norm_linear_position.append(
                df.Hover_replay_linear_position / df.left_well_position
            )
    return np.asarray(norm_linear_position)


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
    axes[0, 0].set_title("Duration")
    axes[0, 0].set_xlim((0, 400))
    sns.despine(ax=axes[0, 0], offset=5)

    # Distance from Animal
    plot_replay_distance_from_actual_position(
        replay_info, kind="box", ax=axes[0, 1], fliersize=fliersize,
        saturation=saturation
    )
    axes[0, 1].set_title("Distance from Animal")
    sns.despine(ax=axes[0, 1], offset=5)
    axes[0, 1].set_xlim((0, 250))
    axes[0, 1].set_yticks([])
    axes[0, 1].spines["left"].set_visible(False)

    # Non-Local Hover Position
    norm_non_local_hover = get_norm_linear_position(replay_info)
    sns.distplot(
        norm_non_local_hover,
        kde_kws=dict(
            bw=0.020,
            clip=(0, 1),
            shade=True,
            facecolor=STATE_COLORS["Hover"],
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

    sns.despine(ax=axes[1, 0], offset=5)
    axes[1, 0].set_xlim((0, 1))
    axes[1, 0].set_title("Non-Local Stationary Position")
    n_non_local = norm_non_local_hover.size
    axes[1, 0].text(0.75, 3.5, f"N = {n_non_local}", zorder=100, fontsize=9)

    # Population firing rate
    _plot_category(
        replay_info,
        "population_rate",
        kind="box",
        ax=axes[1, 1],
        fliersize=fliersize,
        saturation=saturation,
    )
    axes[1, 1].set_xlim((0, 400))
    axes[1, 1].set_xlabel("Rate [spikes / s]")
    axes[1, 1].set_title("Multiunit Population Rate")
    sns.despine(ax=axes[1, 1], offset=5)
    axes[1, 1].set_yticks([])
    axes[1, 1].spines["left"].set_visible(False)
