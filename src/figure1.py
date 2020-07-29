import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from src.parameters import (PAGE_HEIGHT, STATE_COLORS, TRANSITION_TO_CATEGORY,
                            TWO_COLUMN)


def plot_spikes(test_spikes, replay_time, ax=None, cmap=plt.get_cmap("tab20")):
    if ax is None:
        ax = plt.gca()
    spike_ind, neuron_ind = np.nonzero(test_spikes)

    c = [cmap.colors[ind] for ind in neuron_ind]
    ax.eventplot(
        replay_time[spike_ind][:, np.newaxis],
        lineoffsets=neuron_ind + 1,
        colors=c,
        linewidth=2,
    )
    ax.set_yticks((1, test_spikes.shape[1]))
    ax.set_ylabel("Cells")
    ax.set_ylim((0, test_spikes.shape[1] + 0.4))
    ax.set_xlabel("Time [ms]")
    ax.set_xlim((replay_time.min(), replay_time.max() + 1))
    ax.set_xticks((replay_time.min(), replay_time.max() + 1))
    ax.set_title("Synthetic Replay Spike Train", fontsize=16)
    sns.despine()


def plot_place_field_estimates(
        classifier, ax=None, cmap=plt.get_cmap("tab20")):
    if ax is None:
        ax = plt.gca()

    for place_field, color in zip(classifier.place_fields_.T, cmap.colors):
        ax.plot(
            classifier.place_bin_centers_, place_field * 1000, linewidth=2,
            color=color
        )

    ax.set_ylabel("Firing Rate\n[spikes / s]")
    ax.set_ylim([0, np.ceil(classifier.place_fields_.max() * 1000)])
    ax.set_yticks([0, np.ceil(classifier.place_fields_.max() * 1000)])

    ax.set_xlabel("Position [cm]")
    dist_lim = (classifier.place_bin_edges_.min(),
                classifier.place_bin_edges_.max())
    ax.set_xlim(dist_lim)
    ax.set_xticks(dist_lim)
    ax.set_title("Place Field Estimates", fontsize=16)

    sns.despine()


def plot_state_transition(classifier):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(TWO_COLUMN * 0.975, PAGE_HEIGHT / 5),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )

    x, y = np.meshgrid(classifier.place_bin_edges_,
                       classifier.place_bin_edges_)
    state_names = [
        TRANSITION_TO_CATEGORY[transition]
        for transition in np.diag(classifier.continuous_transition_types)
    ]
    STATE_ORDER = ["Hover", "Continuous", "Fragmented"]

    for (ax, name) in zip(axes, STATE_ORDER):
        k = state_names.index(name)
        state_transition = classifier.continuous_state_transition_[k, k]
        h = ax.pcolormesh(
            x,
            y,
            state_transition,
            cmap="Blues",
            vmin=0.0,
            vmax=state_transition.max(),
            rasterized=True,
        )
        ax.set_title(name, fontsize=9)

    cbar = fig.colorbar(
        h, label="Probability", aspect=50, ticks=[0, state_transition.max()]
    )
    cbar.ax.set_yticklabels(["0", "Max"])  # vertically oriented colorbar
    cbar.outline.set_visible(False)
    dist_lim = (classifier.place_bin_edges_.min(),
                classifier.place_bin_edges_.max())
    axes[-1].set_xticks(dist_lim)
    axes[-1].set_yticks(dist_lim)
    axes[1].set_xlabel("Current Position [cm]")
    axes[0].set_ylabel("Previous\nPosition [cm]")
    plt.suptitle("Movement Dynamics Model", y=1.1, fontsize=16)

    sns.despine()


def plot_posteriors(results, linear_distance):
    results.acausal_posterior.plot(
        x="time",
        y="position",
        robust=True,
        vmin=0.0,
        rasterized=True,
        col="state",
        figsize=(TWO_COLUMN, PAGE_HEIGHT / 5),
    )

    ylim = linear_distance.min(), linear_distance.max()
    plt.ylim(ylim)
    plt.yticks(ylim)

    xlim = results.time.min(), results.time.max() + 1
    plt.xlim(xlim)
    plt.xticks(xlim)


def plot_state_transition2(classifier, vmax=0.04):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(TWO_COLUMN * 0.975, PAGE_HEIGHT / 5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    x, y = np.meshgrid(classifier.place_bin_edges_,
                       classifier.place_bin_edges_)
    state_names = [
        TRANSITION_TO_CATEGORY[transition]
        for transition in np.diag(classifier.continuous_transition_types)
    ]
    STATE_ORDER = ["Hover", "Continuous", "Fragmented"]

    for (ax, name) in zip(axes, STATE_ORDER):
        k = state_names.index(name)
        state_transition = classifier.continuous_state_transition_[k, k]
        h = ax.pcolormesh(
            x, y, state_transition, cmap="Blues", vmin=0.0, vmax=vmax,
            rasterized=True,
        )
        ax.set_title(name, fontsize=9)

    cbar = fig.colorbar(h, label="Probability", aspect=40, ticks=[0, vmax])
    cbar.ax.set_yticklabels([0, vmax])  # vertically oriented colorbar
    cbar.outline.set_visible(False)
    dist_lim = (classifier.place_bin_edges_.min(),
                classifier.place_bin_edges_.max())
    axes[-1].set_xticks(dist_lim)
    axes[-1].set_yticks(dist_lim)
    axes[1].set_xlabel("Current Position [cm]")
    axes[0].set_ylabel("Previous\nPosition [cm]")
    plt.suptitle("Dynamics Model", y=1.12, fontsize=16)

    sns.despine()


def plot_probabilities(results, ax=None):
    if ax is None:
        ax = plt.gca()
    probabilities = results.acausal_posterior.sum("position").values

    for probability, state in zip(probabilities.T, results.state.values):
        ax.plot(
            results.time,
            probability,
            linewidth=2,
            label=state,
            color=STATE_COLORS[state],
        )
    xlim = results.time.min(), results.time.max() + 1
    ax.set_xticks(xlim)
    ax.set_xlim(xlim)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Probability")
    ax.set_yticks([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_title("Probability of\nMovement Dynamic", fontsize=16)
    sns.despine()


def plot_posterior(
    classifier, results, linear_distance, ax=None, cbar_orientation="vertical",
    cbar_aspect=50,
):
    if ax is None:
        ax = plt.gca()

    posterior = results.acausal_posterior.sum("state").values.T

    t, p = np.meshgrid(np.arange(results.time.size + 1),
                       classifier.place_bin_edges_)

    h = ax.pcolormesh(
        t,
        p,
        posterior,
        vmin=0.0,
        vmax=np.percentile(posterior, 97.5),
        rasterized=True,
        cmap="viridis",
    )
    cbar = plt.colorbar(
        h,
        label="Posterior",
        aspect=cbar_aspect,
        ticks=[0, np.percentile(posterior, 97.5)],
        ax=ax,
        orientation=cbar_orientation,
    )
    cbar.ax.set_yticklabels(["0", "Max"])  # vertically oriented colorbar
    cbar.outline.set_visible(False)

    ax.set_title("Estimate of\nLatent Position", fontsize=16)

    xlim = results.time.min(), results.time.max() + 1
    ax.set_xticks(xlim)
    ax.set_xlim(xlim)
    ax.set_xlabel("Time [ms]")

    ylim = linear_distance.min(), linear_distance.max()
    ax.set_yticks(ylim)
    ax.set_ylim(ylim)
    ax.set_ylabel("Position [cm]")


def plot_classification_of_speeds(probabilities, is_classified, ax=None):
    if ax is None:
        ax = plt.gca()

    for state, prob in zip(probabilities.state.values, probabilities.values.T):
        ax.plot(
            probabilities.Speed,
            prob,
            linewidth=2,
            label=state,
            color=STATE_COLORS[state],
        )

    for state, is_class in zip(is_classified.state.values,
                               is_classified.values.T):
        ax.fill_between(
            probabilities.Speed + 0.01,
            is_class,
            where=is_class.astype(bool),
            alpha=0.25,
            color=STATE_COLORS[state],
        )
        if state != "Fragmented-Continuous-Mix":
            ax.text(
                np.median(probabilities.Speed[is_class.astype(bool)]),
                1.01,
                state,
                va="bottom",
                ha="center",
                fontsize=9,
                color=STATE_COLORS[state],
            )

    ax.set_ylabel("Probability")
    ax.set_ylim((0, 1.01))

    ax.set_xlabel("Speed [cm / s]")
    ax.set_xscale("log")
    ax.set_xticks((1e0, 1e1, 1e2, 1e3, 1e4))
    ax.set_xlim((1e0, 1e4))
    ax.set_title("Classification of Speeds by Movement Dynamic Probability",
                 fontsize=16, pad=10)

    ax.set_xlim((probabilities.Speed.min(), probabilities.Speed.max()))
    sns.despine()
