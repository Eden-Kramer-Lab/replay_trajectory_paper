#!/bin/bash
NCORES=16
NWORKERS=16
CLUSTERLESS_WALLTIME='48:00:00'
REMY_WALLTIME='72:00:00'

# Clusterless
python queue_cluster_jobs_all_times.py --data_type 'clusterless' \
                                       --n_cores $NCORES \
                                       --n_workers $NWORKERS \
                                       --wall_time $CLUSTERLESS_WALLTIME \
                                       --plot_figures

python queue_cluster_jobs_all_times.py --Animal 'remy' --Day 35 --Epoch 2 \
                                       --data_type 'clusterless' \
                                       --n_cores $NCORES \
                                       --n_workers $NWORKERS \
                                       --wall_time $REMY_WALLTIME \
                                       --plot_figures \

python queue_cluster_jobs_all_times.py --Animal 'remy' --Day 35 --Epoch 4 \
                                       --data_type 'clusterless' \
                                       --n_cores $NCORES \
                                       --n_workers $NWORKERS \
                                       --wall_time $REMY_WALLTIME \
                                       --plot_figures \

python queue_cluster_jobs_all_times.py --Animal 'remy' --Day 36 --Epoch 2 \
                                       --data_type 'clusterless' \
                                       --n_cores $NCORES \
                                       --n_workers $NWORKERS \
                                       --wall_time $REMY_WALLTIME \
                                       --plot_figures \

python queue_cluster_jobs_all_times.py --Animal 'remy' --Day 36 --Epoch 4 \
                                       --data_type 'clusterless' \
                                       --n_cores $NCORES \
                                       --n_workers $NWORKERS \
                                       --wall_time $REMY_WALLTIME \
                                       --plot_figures \

python queue_cluster_jobs_all_times.py --Animal 'remy' --Day 37 --Epoch 2 \
                                       --data_type 'clusterless' \
                                       --n_cores $NCORES \
                                       --n_workers $NWORKERS \
                                       --wall_time $REMY_WALLTIME \
                                       --plot_figures \

python queue_cluster_jobs_all_times.py --Animal 'remy' --Day 37 --Epoch 4 \
                                       --data_type 'clusterless' \
                                       --n_cores $NCORES \
                                       --n_workers $NWORKERS \
                                       --wall_time $REMY_WALLTIME \
                                       --plot_figures \
