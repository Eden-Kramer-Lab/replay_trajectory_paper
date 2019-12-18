from os.path import abspath, dirname, join, pardir

import numpy as np
from loren_frank_data_processing import Animal
from replay_trajectory_classification.misc import NumbaKDE

# LFP sampling frequency
SAMPLING_FREQUENCY = 500

# Data directories and definitions
ROOT_DIR = join(abspath(dirname(__file__)), pardir)
RAW_DATA_DIR = join(ROOT_DIR, 'Raw-Data')
PROCESSED_DATA_DIR = join(ROOT_DIR, 'Processed-Data')
FIGURE_DIR = join(ROOT_DIR, 'figures')

ANIMALS = {
    'bon': Animal(directory=join(RAW_DATA_DIR, 'Bond'), short_name='bon'),
    'cha': Animal(directory=join(RAW_DATA_DIR, 'Chapati'), short_name='cha'),
    'con': Animal(directory=join(RAW_DATA_DIR, 'Conley'), short_name='con'),
    'Cor': Animal(directory=join(RAW_DATA_DIR, 'Corriander'),
                  short_name='Cor'),
    'dav': Animal(directory=join(RAW_DATA_DIR, 'Dave'), short_name='dav'),
    'dud': Animal(directory=join(RAW_DATA_DIR, 'Dudley'), short_name='dud'),
    'egy': Animal(directory=join(RAW_DATA_DIR, 'Egypt'), short_name='egy'),
    'fra': Animal(directory=join(RAW_DATA_DIR, 'Frank'), short_name='fra'),
    'gov': Animal(directory=join(RAW_DATA_DIR, 'Government'),
                  short_name='gov'),
    'hig': Animal(directory=join(RAW_DATA_DIR, 'Higgs'), short_name='hig'),
    'remy': Animal(directory=join(RAW_DATA_DIR, 'Remy'), short_name='remy'),
}

_MARKS = ['channel_1_max', 'channel_2_max', 'channel_3_max', 'channel_4_max']
_BRAIN_AREAS = ['CA1', 'CA2', 'CA3']

# Classifier Parameters
place_bin_size = 2.0
# 10 m/s = 2 cm / 2 ms
movement_var = 4.0
replay_speed = 1
model = NumbaKDE
model_kwargs = {
    'bandwidth': np.array([24.0, 24.0, 24.0, 24.0, 5.0, 5.0])
}
knot_spacing = 5
spike_model_penalty = 0.5
discrete_diag = 1 - 1E-3
continuous_transition_types = (
    [['w_track_1D_random_walk_minus_identity', 'w_track_1D_inverse_random_walk', 'identity'],  # noqa
     ['uniform',                               'w_track_1D_inverse_random_walk', 'uniform'],   # noqa
     ['w_track_1D_random_walk_minus_identity', 'w_track_1D_inverse_random_walk', 'identity']])  # noqa

TRANSITION_TO_CATEGORY = {
    'identity': 'Hover',
    'uniform': 'Fragmented',
    'w_track_1D_inverse_random_walk': 'Fragmented',
    'inverse_random_walk': 'Fragmented',
    'random_walk': 'Continuous',
    'random_walk_minus_identity': 'Continuous',
    'w_track_1D_random_walk': 'Continuous',
    'w_track_1D_random_walk_minus_identity': 'Continuous',
}

PROBABILITY_THRESHOLD = 0.8

STATE_ORDER = ['Hover', 'Hover-Continuous-Mix', 'Continuous',
               'Fragmented-Continuous-Mix', 'Fragmented']

# Plotting Colors
STATE_COLORS = {
    'Hover': '#9f043a',
    'Fragmented': '#ff6944',
    'Continuous': '#521b65',
    'Hover-Continuous-Mix': '#61c5e6',
    'Fragmented-Continuous-Mix': '#2a586a',
    '': '#c7c7c7',
}
