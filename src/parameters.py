from os.path import abspath, dirname, join, pardir

import numpy as np
from replay_trajectory_classification.misc import NumbaKDE

from loren_frank_data_processing import Animal

# LFP sampling frequency
SAMPLING_FREQUENCY = 1000

# Data directories and definitions
ROOT_DIR = join(abspath(dirname(__file__)), pardir)
RAW_DATA_DIR = join(ROOT_DIR, 'Raw-Data')
PROCESSED_DATA_DIR = join(ROOT_DIR, 'Processed-Data')
FIGURE_DIR = join(ROOT_DIR, 'figures')

ANIMALS = {
    'bon': Animal(directory=join(RAW_DATA_DIR, 'Bond'), short_name='bon'),
}

_MARKS = ['channel_1_max', 'channel_2_max', 'channel_3_max', 'channel_4_max']
_BRAIN_AREAS = ['CA1', 'CA3']


place_bin_size = 2.0
movement_var = 1.0
replay_speed = 1
model = NumbaKDE
model_kwargs = {
    'bandwidth': np.array([24.0, 24.0, 24.0, 24.0, 6.0, 6.0])
    }
