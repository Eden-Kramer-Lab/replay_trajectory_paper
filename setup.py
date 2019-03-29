#!/usr/bin/env python3

from setuptools import find_packages, setup

INSTALL_REQUIRES = []
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
