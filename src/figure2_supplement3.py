import logging

import xarray as xr
from loren_frank_data_processing.position import (EDGE_ORDER, EDGE_SPACING,
                                                  make_track_graph)
from replay_trajectory_classification import ClusterlessClassifier
from sklearn.model_selection import KFold
from src.load_data import load_data
from src.parameters import (ANIMALS, TRANSITION_TO_CATEGORY,
                            continuous_transition_types, discrete_diag, model,
                            model_kwargs, movement_var, place_bin_size,
                            replay_speed)
from src.visualization import plot_run_slice

logging.basicConfig(
    level='INFO', format='%(asctime)s %(message)s',
    datefmt='%d-%b-%y %H:%M:%S')


def main():
    epoch_key = 'bon', 5, 2

    data = load_data(epoch_key)
    track_graph, center_well_id = make_track_graph(epoch_key, ANIMALS)

    cv = KFold()
    results = []

    for fold_ind, (train, test) in enumerate(
            cv.split(data["position_info"].index)):
        logging.info(f'Fitting Fold #{fold_ind + 1}...')
        classifier = ClusterlessClassifier(
            place_bin_size=place_bin_size,
            movement_var=movement_var,
            replay_speed=replay_speed,
            discrete_transition_diag=discrete_diag,
            continuous_transition_types=continuous_transition_types,
            model=model,
            model_kwargs=model_kwargs)
        classifier.fit(
            position=data["position_info"].iloc[train].linear_position,
            multiunits=data["multiunit"].isel(time=train),
            track_graph=track_graph,
            center_well_id=center_well_id,
            edge_order=EDGE_ORDER,
            edge_spacing=EDGE_SPACING,
        )

        logging.info('Predicting posterior...')
        results.append(
            classifier.predict(
                data["multiunit"].isel(time=test),
                time=data["position_info"].iloc[test].index,
            )
        )

    results = (xr.concat(results, dim="time")
               .assign_coords(state=lambda ds:
                              ds.state.to_index().map(TRANSITION_TO_CATEGORY)))

    start_time, end_time = data['position_info'].index[[340_000, 346_000]]

    plot_run_slice(results, data, classifier, start_time, end_time)


if __name__ == "__main__":
    main()
