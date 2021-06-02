#!/bin/bash
REMY_WALLTIME='24:00:00'

python queue_cluster_jobs_standard_decoder.py

python queue_cluster_jobs_standard_decoder.py --Animal 'remy' --Day 35 --Epoch 2 \
                                              --wall_time $REMY_WALLTIME \

python queue_cluster_jobs_standard_decoder.py --Animal 'remy' --Day 35 --Epoch 4 \
                                              --wall_time $REMY_WALLTIME \

python queue_cluster_jobs_standard_decoder.py --Animal 'remy' --Day 36 --Epoch 2 \
                                              --wall_time $REMY_WALLTIME \

python queue_cluster_jobs_standard_decoder.py --Animal 'remy' --Day 36 --Epoch 4 \
                                              --wall_time $REMY_WALLTIME \

python queue_cluster_jobs_standard_decoder.py --Animal 'remy' --Day 37 --Epoch 2 \
                                              --wall_time $REMY_WALLTIME \

python queue_cluster_jobs_standard_decoder.py --Animal 'remy' --Day 37 --Epoch 4 \
                                              --wall_time $REMY_WALLTIME \
