#!/bin/bash
NCORES=16
NWORKERS=1
WALLTIME='12:00:00'

python queue_cluster_jobs.py --data_type 'sorted_spikes' --n_cores $NCORES --n_workers $NWORKERS --wall_time $WALLTIME
python queue_cluster_jobs.py --data_type 'clusterless' --n_cores $NCORES --n_workers $NWORKERS --wall_time $WALLTIME
python queue_cluster_jobs.py --Animal 'remy' --Day 35 --Epoch 2 --data_type 'clusterless' --n_cores $NCORES --n_workers $NWORKERS --wall_time $WALLTIME
python queue_cluster_jobs.py --Animal 'remy' --Day 35 --Epoch 4 --data_type 'clusterless' --n_cores $NCORES --n_workers $NWORKERS --wall_time $WALLTIME
python queue_cluster_jobs.py --Animal 'remy' --Day 36 --Epoch 2 --data_type 'clusterless' --n_cores $NCORES --n_workers $NWORKERS --wall_time $WALLTIME
python queue_cluster_jobs.py --Animal 'remy' --Day 36 --Epoch 4 --data_type 'clusterless' --n_cores $NCORES --n_workers $NWORKERS --wall_time $WALLTIME
python queue_cluster_jobs.py --Animal 'remy' --Day 37 --Epoch 2 --data_type 'clusterless' --n_cores $NCORES --n_workers $NWORKERS --wall_time $WALLTIME
python queue_cluster_jobs.py --Animal 'remy' --Day 37 --Epoch 4 --data_type 'clusterless' --n_cores $NCORES --n_workers $NWORKERS --wall_time $WALLTIME


# python queue_cluster_jobs.py --Animal 'fra' --Day 5 --Epoch 6 --data_type 'sorted_spikes' --n_cores 1 --n_workers 1 --wall_time '01:00:00'
# python queue_cluster_jobs.py --Animal 'Cor' --Day 4 --Epoch 2 --data_type 'sorted_spikes' --n_cores 1 --n_workers 1 --wall_time '01:00:00'
