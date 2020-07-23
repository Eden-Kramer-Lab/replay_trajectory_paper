import os
from glob import glob

import pandas as pd
from src.parameters import PROCESSED_DATA_DIR, STATE_ORDER


def main():
    data_type, dim = "clusterless", "1D"
    file_paths = glob(
        os.path.join(PROCESSED_DATA_DIR,
                     f"*_{data_type}_{dim}_replay_info.csv")
    )

    for file_path in file_paths:
        replay_info = pd.read_csv(file_path).set_index(
            ["animal", "day", "epoch", "ripple_number"]
        )
        replay_info = replay_info.rename(
            columns={
                f"{state}_popultion_rate": f"{state}_population_rate"
                for state in STATE_ORDER
            }
        )
        animal, day, epoch = replay_info.index[0][:3]
        epoch_identifier = f"{animal}_{day:02d}_{epoch:02d}_{data_type}_{dim}"
        replay_info_filename = os.path.join(
            PROCESSED_DATA_DIR, f"{epoch_identifier}_replay_info.csv"
        )
        replay_info.to_csv(replay_info_filename)


if __name__ == '__main__':
    main()
