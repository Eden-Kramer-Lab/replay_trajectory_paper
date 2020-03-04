#!/bin/bash
NCORES=16
NWORKERS=16
SORTED_WALLTIME='5:00:00'
CLUSTERLESS_WALLTIME='12:00:00'
REMY_WALLTIME='24:00:00'

# Clusterless
python queue_cluster_jobs.py --data_type 'clusterless' --n_cores $NCORES --n_workers $NWORKERS --wall_time $CLUSTERLESS_WALLTIME --plot_ripple_figures
python queue_cluster_jobs.py --Animal 'remy' --Day 35 --Epoch 2 --data_type 'clusterless' --n_cores $NCORES --n_workers $NWORKERS --wall_time $REMY_WALLTIME --plot_ripple_figures
python queue_cluster_jobs.py --Animal 'remy' --Day 35 --Epoch 4 --data_type 'clusterless' --n_cores $NCORES --n_workers $NWORKERS --wall_time $REMY_WALLTIME --plot_ripple_figures
python queue_cluster_jobs.py --Animal 'remy' --Day 36 --Epoch 2 --data_type 'clusterless' --n_cores $NCORES --n_workers $NWORKERS --wall_time $REMY_WALLTIME --plot_ripple_figures
python queue_cluster_jobs.py --Animal 'remy' --Day 36 --Epoch 4 --data_type 'clusterless' --n_cores $NCORES --n_workers $NWORKERS --wall_time $REMY_WALLTIME --plot_ripple_figures
python queue_cluster_jobs.py --Animal 'remy' --Day 37 --Epoch 2 --data_type 'clusterless' --n_cores $NCORES --n_workers $NWORKERS --wall_time $REMY_WALLTIME --plot_ripple_figures
python queue_cluster_jobs.py --Animal 'remy' --Day 37 --Epoch 4 --data_type 'clusterless' --n_cores $NCORES --n_workers $NWORKERS --wall_time $REMY_WALLTIME --plot_ripple_figures

# Sorted Spikes
python queue_cluster_jobs.py --data_type 'sorted_spikes' --n_cores $NCORES --n_workers $NWORKERS --wall_time $SORTED_WALLTIME --plot_ripple_figures
