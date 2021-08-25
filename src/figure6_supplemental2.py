import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr
from replay_trajectory_classification import SortedSpikesClassifier
from src.analysis import (get_is_classified, get_place_field_max,
                          get_probability)
from src.figure_utilities import (PAGE_HEIGHT, TWO_COLUMN, save_figure,
                                  set_figure_defaults)
from src.parameters import (FIGURE_DIR, PROBABILITY_THRESHOLD,
                            PROCESSED_DATA_DIR, STATE_COLORS)
from src.standard_decoder import (fit_mark_likelihood, load_data,
                                  normalize_to_posterior,
                                  predict_clusterless_wtrack,
                                  predict_mark_likelihood,
                                  predict_poisson_likelihood)


def fit_and_load_models(epoch_key):
    # clusterless model
    data_type, dim = "clusterless", "1D"
    animal, day, epoch = epoch_key

    clusterless_state_space_results = xr.open_dataset(
        os.path.join(PROCESSED_DATA_DIR, f"{animal}_{day:02}_{epoch:02}.nc"),
        group=f"/{data_type}/{dim}/classifier/ripples/",
    )

    # sorted spikes model
    data_type, dim = "sorted_spikes", "1D"
    animal, day, epoch = epoch_key

    sorted_spikes_state_space_results = xr.open_dataset(
        os.path.join(PROCESSED_DATA_DIR, f"{animal}_{day:02}_{epoch:02}.nc"),
        group=f"/{data_type}/{dim}/classifier/ripples/",
    )

    model_name = os.path.join(
        PROCESSED_DATA_DIR,
        f"{animal}_{day:02}_{epoch:02}_{data_type}_{dim}_model.pkl"
    )
    sorted_spikes_classifier = SortedSpikesClassifier.load_model(model_name)

    (
        linear_position,
        multiunit_dfs,
        position_info,
        ripple_times,
        track_graph,
        center_well_id,
        spike_times,
    ) = load_data(epoch_key)

    (
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
    ) = fit_mark_likelihood(
        linear_position, multiunit_dfs, track_graph, center_well_id,
        dt=0.002)

    place_fields = np.asarray(sorted_spikes_classifier.place_fields_)

    place_field_max = get_place_field_max(sorted_spikes_classifier)
    linear_position_order = place_field_max.argsort(axis=0).squeeze()

    ordered_spike_times = [spike_times[i] for i in linear_position_order]

    return (clusterless_state_space_results, sorted_spikes_state_space_results,
            place_bin_centers, occupancy, joint_pdf_models, multiunit_dfs,
            ground_process_intensities, mean_rates, is_track_interior,
            place_bin_edges, track_graph1, place_bin_center_ind_to_node,
            spike_times, is_track_interior, place_fields, position_info,
            ripple_times, ordered_spike_times, multiunit_dfs)


def get_clusterless_posteriors_and_fits(
        start_time, end_time, place_bin_centers, occupancy,
        joint_pdf_models, multiunit_dfs, ground_process_intensities,
        mean_rates, is_track_interior, place_bin_edges,
        track_graph1, place_bin_center_ind_to_node, dt):
    clusterless_likelihood, clusterless_time = predict_mark_likelihood(
        start_time,
        end_time,
        place_bin_centers,
        occupancy,
        joint_pdf_models,
        multiunit_dfs,
        ground_process_intensities,
        mean_rates,
        is_track_interior,
        dt
    )

    (
        clusterless_time,
        clusterless_likelihood,
        clusterless_radon_speed,
        clusterless_radon_prediction,
        clusterless_radon_score,
        clusterless_isotonic_speed,
        clusterless_isotonic_prediction,
        clusterless_isotonic_score,
        clusterless_linear_speed,
        clusterless_linear_prediction,
        clusterless_linear_score,
        clusterless_map_speed,
        clusterless_map_prediction,
        clusterless_map_score,
    ) = predict_clusterless_wtrack(
        clusterless_time,
        clusterless_likelihood,
        place_bin_centers,
        is_track_interior,
        place_bin_edges,
        track_graph1,
        place_bin_center_ind_to_node,
        dt
    )

    clusterless_posterior = xr.DataArray(
        data=normalize_to_posterior(clusterless_likelihood),
        dims=['time', 'position'],
        coords={'time': clusterless_time,
                'position': place_bin_centers.squeeze()}
    )

    return (clusterless_posterior,
            clusterless_time,
            clusterless_likelihood,
            clusterless_radon_speed,
            clusterless_radon_prediction,
            clusterless_radon_score,
            clusterless_isotonic_speed,
            clusterless_isotonic_prediction,
            clusterless_isotonic_score,
            clusterless_linear_speed,
            clusterless_linear_prediction,
            clusterless_linear_score,
            clusterless_map_speed,
            clusterless_map_prediction,
            clusterless_map_score)


def get_sorted_spikes_posteriors_and_fits(
        start_time, end_time, place_fields, spike_times, is_track_interior,
        place_bin_centers, place_bin_edges, track_graph1,
        place_bin_center_ind_to_node, dt):
    sorted_spikes_likelihood, sorted_spikes_time = predict_poisson_likelihood(
        start_time, end_time, spike_times, place_fields, is_track_interior,
        dt)

    (
        sorted_spikes_time,
        sorted_spikes_likelihood,
        sorted_spikes_radon_speed,
        sorted_spikes_radon_prediction,
        sorted_spikes_radon_score,
        sorted_spikes_isotonic_speed,
        sorted_spikes_isotonic_prediction,
        sorted_spikes_isotonic_score,
        sorted_spikes_linear_speed,
        sorted_spikes_linear_prediction,
        sorted_spikes_linear_score,
        sorted_spikes_map_speed,
        sorted_spikes_map_prediction,
        sorted_spikes_map_score,
    ) = predict_clusterless_wtrack(
        sorted_spikes_time,
        sorted_spikes_likelihood,
        place_bin_centers,
        is_track_interior,
        place_bin_edges,
        track_graph1,
        place_bin_center_ind_to_node,
        dt
    )

    sorted_spikes_posterior = xr.DataArray(
        data=normalize_to_posterior(sorted_spikes_likelihood),
        dims=['time', 'position'],
        coords={'time': sorted_spikes_time - start_time,
                'position': place_bin_centers.squeeze()}
    )

    return (sorted_spikes_posterior,
            sorted_spikes_time,
            sorted_spikes_likelihood,
            sorted_spikes_radon_speed,
            sorted_spikes_radon_prediction,
            sorted_spikes_radon_score,
            sorted_spikes_isotonic_speed,
            sorted_spikes_isotonic_prediction,
            sorted_spikes_isotonic_score,
            sorted_spikes_linear_speed,
            sorted_spikes_linear_prediction,
            sorted_spikes_linear_score,
            sorted_spikes_map_speed,
            sorted_spikes_map_prediction,
            sorted_spikes_map_score,
            )


def plot_posteriors(ripple_number, start_time, end_time, position_info,
                    spike_times, is_track_interior, sorted_spikes_posterior,
                    clusterless_posterior, sorted_spikes_radon_prediction,
                    sorted_spikes_map_prediction, clusterless_radon_prediction,
                    clusterless_map_prediction,
                    clusterless_state_space_results, multiunit_spike_times):
    cmap = copy.copy(plt.cm.get_cmap('bone_r'))
    cmap.set_bad(color="lightgrey", alpha=1.0)

    max_position = int(np.ceil(position_info.linear_position.max()))
    linear_position = np.asarray(
        position_info
        .set_index(position_info.index / np.timedelta64(1, 's'))
        .loc[start_time:end_time, 'linear_position'])[0]

    MILLISECONDS_TO_SECONDS = 1000

    fig, axes = plt.subplots(
        6, 1, figsize=(TWO_COLUMN / 3, PAGE_HEIGHT / 2),
        constrained_layout=True,
        sharex=False, sharey=False,
        gridspec_kw={"height_ratios": [2, 3, 2, 3, 3, 1]},)

    # ax 0 - Sorted Spikes
    axes[0].eventplot(
        [MILLISECONDS_TO_SECONDS * (times[(times >= start_time) &
                                          (times <= end_time)] - start_time)
            for times in spike_times], color="black", clip_on=False)
    axes[0].set_ylim((-0.5, len(spike_times) - 0.5))
    axes[0].set_yticks((0, len(spike_times) - 1))
    axes[0].set_yticklabels((1, len(spike_times)))
    axes[0].set_ylabel('Cells')
    axes[0].set_xticks([])
    axes[0].set_xlim((0.0, MILLISECONDS_TO_SECONDS * (end_time - start_time)))
    sns.despine(ax=axes[0], offset=5)
    axes[0].spines["bottom"].set_visible(False)

    # ax 1 - Standard Decoder, Sorted Spikes
    (sorted_spikes_posterior
     .assign_coords(time=lambda ds: ds.time * MILLISECONDS_TO_SECONDS)
     .where(is_track_interior)
     .plot(x='time', y='position', ax=axes[1], add_colorbar=False, cmap=cmap,
           robust=True))
    axes[1].set_xlabel('')
    axes[1].set_title(
        'Sorted Spikes Standard Decoder, 2 ms bins', fontsize=7)
    axes[1].set_ylabel('Pos. [cm]')
    axes[1].plot(sorted_spikes_posterior.time *
                 MILLISECONDS_TO_SECONDS, sorted_spikes_radon_prediction,
                 color='#1f77b4')
    axes[1].plot(sorted_spikes_posterior.time *
                 MILLISECONDS_TO_SECONDS, sorted_spikes_map_prediction,
                 color='#2ca02c')
    axes[1].axhline(linear_position, color='magenta',
                    linestyle='--', zorder=100, linewidth=2)
    axes[1].set_ylim((0, max_position))
    axes[1].set_yticks((0, max_position))
    sns.despine(offset=5, ax=axes[1])
    axes[1].set_xlim((0.0, MILLISECONDS_TO_SECONDS * (end_time - start_time)))
    axes[1].set_xticks([])
    sns.despine(ax=axes[1], offset=5)
    axes[1].spines["bottom"].set_visible(False)

    # ax 2 - Multiunit Spikes
    axes[2].eventplot(
        [MILLISECONDS_TO_SECONDS * (times[(times >= start_time) &
                                          (times <= end_time)] - start_time)
            for times in multiunit_spike_times], color="black", clip_on=False)
    axes[2].set_ylim((-0.5, len(multiunit_spike_times) - 0.5))
    axes[2].set_yticks((0, len(multiunit_spike_times) - 1))
    axes[2].set_yticklabels((1, len(multiunit_spike_times)))
    axes[2].set_ylabel('Tet.')
    axes[2].set_xticks([])
    axes[2].set_xlim((0.0, MILLISECONDS_TO_SECONDS * (end_time - start_time)))
    sns.despine(ax=axes[2], offset=5)
    axes[2].spines["bottom"].set_visible(False)

    # ax 3 - Standard Decoder, Clusterless
    (clusterless_posterior
     .assign_coords(time=lambda ds: ds.time * MILLISECONDS_TO_SECONDS)
     .where(is_track_interior)
     .plot(x='time', y='position', ax=axes[3], add_colorbar=False, cmap=cmap,
           robust=True))
    axes[3].set_xlabel('')
    axes[3].set_title('Clusterless Standard Decoder, 2 ms bins', fontsize=7)
    axes[3].set_ylabel('Pos. [cm]')
    axes[3].plot(clusterless_posterior.time *
                 MILLISECONDS_TO_SECONDS, clusterless_radon_prediction,
                 color='#1f77b4')
    axes[3].plot(clusterless_posterior.time *
                 MILLISECONDS_TO_SECONDS, clusterless_map_prediction,
                 color='#2ca02c')
    axes[3].axhline(linear_position, color='magenta',
                    linestyle='--', zorder=100, linewidth=2)

    axes[3].set_ylim((0, max_position))
    axes[3].set_yticks((0, max_position))
    axes[3].set_xlim((0.0, MILLISECONDS_TO_SECONDS * (end_time - start_time)))
    axes[3].set_xticks([])
    sns.despine(ax=axes[3], offset=5)
    axes[3].spines["bottom"].set_visible(False)

    # ax 4 - Clusterless State Space
    (clusterless_state_space_results
     .sel(ripple_number=ripple_number)
     .acausal_posterior
     .dropna('time', how='all')
     .assign_coords(time=lambda ds: ds.time * MILLISECONDS_TO_SECONDS
                    / np.timedelta64(1, 's'))
     .sum('state')
     .where(is_track_interior)
     .plot(x='time', y='position', robust=True, ax=axes[4], add_colorbar=False,
           cmap=cmap))
    axes[4].axhline(linear_position, color='magenta',
                    linestyle='--', zorder=100, linewidth=2)
    axes[4].set_title('Clusterless State Space, 2 ms bins', fontsize=7)
    axes[4].set_xlabel('')
    axes[4].set_ylabel('Pos. [cm]')

    axes[4].set_ylim((0, max_position))
    axes[4].set_yticks((0, max_position))
    axes[4].set_xlim((0.0, MILLISECONDS_TO_SECONDS * (end_time - start_time)))
    axes[4].set_xticks([])
    sns.despine(ax=axes[4], offset=5)
    axes[4].spines["bottom"].set_visible(False)

    # ax 5 Probability of Dynamic
    probability = get_probability(clusterless_state_space_results.sel(
        ripple_number=ripple_number).dropna('time', how='all'))
    is_classified = get_is_classified(probability, PROBABILITY_THRESHOLD)

    for state, prob in zip(clusterless_state_space_results.state.values,
                           probability.values.T):
        axes[5].plot(
            MILLISECONDS_TO_SECONDS *
            probability.time / np.timedelta64(1, "s"),
            prob,
            linewidth=1,
            color=STATE_COLORS[state],
            clip_on=False,
        )

    for state, is_class in zip(is_classified.state.values,
                               is_classified.values.T):
        if is_class.sum() > 0:
            axes[5].fill_between(
                MILLISECONDS_TO_SECONDS *
                probability.time / np.timedelta64(1, 's'),
                is_class,
                where=is_class.astype(bool),
                alpha=0.25,
                color=STATE_COLORS[state],
            )
    axes[5].set_ylim((0, 1))
    axes[5].set_title('')
    axes[5].set_ylabel('Prob.')
    sns.despine(offset=5, ax=axes[5])
    axes[5].set_xlim((0.0, MILLISECONDS_TO_SECONDS * (end_time - start_time)))

    axes[-1].set_xlabel('Time [ms]')


def plot_figure(epoch_key, ripple_numbers, is_save_figure=False):
    if isinstance(ripple_numbers, int):
        ripple_numbers = [ripple_numbers]

    set_figure_defaults()
    (clusterless_state_space_results, _,
     place_bin_centers, occupancy, joint_pdf_models, multiunit_dfs,
     ground_process_intensities, mean_rates, is_track_interior,
     place_bin_edges, track_graph1, place_bin_center_ind_to_node,
     spike_times, is_track_interior, place_fields, position_info,
     ripple_times, ordered_spike_times, multiunit_dfs
     ) = fit_and_load_models(epoch_key)

    for ripple_number in ripple_numbers:
        start_time, end_time = (
            ripple_times.loc[ripple_number].start_time /
            np.timedelta64(1, "s"),
            ripple_times.loc[ripple_number].end_time /
            np.timedelta64(1, "s"),
        )
        (clusterless_posterior,
         clusterless_time,
         clusterless_likelihood,
         clusterless_radon_speed,
         clusterless_radon_prediction,
         clusterless_radon_score,
         clusterless_isotonic_speed,
         clusterless_isotonic_prediction,
         clusterless_isotonic_score,
         clusterless_linear_speed,
         clusterless_linear_prediction,
         clusterless_linear_score,
         clusterless_map_speed,
         clusterless_map_prediction,
         clusterless_map_score) = get_clusterless_posteriors_and_fits(
            start_time, end_time, place_bin_centers, occupancy,
            joint_pdf_models, multiunit_dfs, ground_process_intensities,
            mean_rates, is_track_interior, place_bin_edges, track_graph1,
            place_bin_center_ind_to_node, dt=0.002)

        (sorted_spikes_posterior,
         sorted_spikes_time,
         sorted_spikes_likelihood,
         sorted_spikes_radon_speed,
         sorted_spikes_radon_prediction,
         sorted_spikes_radon_score,
         sorted_spikes_isotonic_speed,
         sorted_spikes_isotonic_prediction,
         sorted_spikes_isotonic_score,
         sorted_spikes_linear_speed,
         sorted_spikes_linear_prediction,
         sorted_spikes_linear_score,
         sorted_spikes_map_speed,
         sorted_spikes_map_prediction,
         sorted_spikes_map_score,
         ) = get_sorted_spikes_posteriors_and_fits(
            start_time, end_time, place_fields, spike_times,
            is_track_interior, place_bin_centers, place_bin_edges,
            track_graph1, place_bin_center_ind_to_node, dt=0.002)

        multiunit_spike_times = [
            df.index / np.timedelta64(1, 's') for df in multiunit_dfs]
        plot_posteriors(
            ripple_number, start_time, end_time, position_info,
            ordered_spike_times, is_track_interior, sorted_spikes_posterior,
            clusterless_posterior, sorted_spikes_radon_prediction,
            sorted_spikes_map_prediction, clusterless_radon_prediction,
            clusterless_map_prediction, clusterless_state_space_results,
            multiunit_spike_times)

        if is_save_figure:
            figure_dir = os.path.join(FIGURE_DIR, "Figure6-supplemental2")
            os.makedirs(figure_dir, exist_ok=True)
            animal, day, epoch = epoch_key
            fig_name = (
                "figure6_supplemental2_"
                f"{animal}_{day:02d}_{epoch:02d}_{ripple_number:04d}_"
                f"decoder_comparison"
            )
            save_figure(os.path.join(figure_dir, fig_name))


if __name__ == '__main__':
    plot_figure(epoch_key=('bon', 4, 2),
                ripple_numbers=158,
                is_save_figure=True)
    plot_figure(epoch_key=('fra', 6, 6),
                ripple_numbers=151,
                is_save_figure=True)
    plot_figure(epoch_key=('Cor', 1, 4),
                ripple_numbers=84,
                is_save_figure=True)
    plot_figure(epoch_key=('bon', 3, 4),
                ripple_numbers=12,
                is_save_figure=True)
    plot_figure(epoch_key=('bon', 3, 6),
                ripple_numbers=[5, 93],
                is_save_figure=True)
