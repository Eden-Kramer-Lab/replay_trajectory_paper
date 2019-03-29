#!/usr/bin/env python3

from setuptools import find_packages, setup

INSTALL_REQUIRES = ['numpy >= 1.15', 'matplotlib >= 3.0', 'pandas >= 0.21',
                    'xarray', 'scipy', 'seaborn', 'ripple_detection',
                    'loren_frank_data_processing',
                    'replay_trajectory_classification']
TESTS_REQUIRE = ['pytest >= 2.7.1']

setup(
    name='replay_trajectory_paper',
    version='0.1.0.dev0',
    license='MIT',
    description=('Classify replay trajectories.'),
    author='Eric Denovellis',
    author_email='edeno@bu.edu',
    url='https://github.com/Eden-Kramer-Lab/replay_trajectory_paper',
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
)
