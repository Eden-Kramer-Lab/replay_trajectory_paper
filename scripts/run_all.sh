#!/bin/bash
python queue_cluster_jobs.py --data_type 'sorted_spikes' --n_cores 1 --n_workers 1 --wall_time '00:30:00'
python queue_cluster_jobs.py --data_type 'clusterless' --n_cores 1 --n_workers 1 --wall_time '00:30:00'
python queue_cluster_jobs.py --Animal 'remy' --Day 35 --Epoch 2 --data_type 'clusterless' --n_cores 1 --n_workers 1 --wall_time '00:30:00'
python queue_cluster_jobs.py --Animal 'remy' --Day 35 --Epoch 4 --data_type 'clusterless' --n_cores 1 --n_workers 1 --wall_time '00:30:00'
python queue_cluster_jobs.py --Animal 'remy' --Day 36 --Epoch 2 --data_type 'clusterless' --n_cores 1 --n_workers 1 --wall_time '00:30:00'
python queue_cluster_jobs.py --Animal 'remy' --Day 36 --Epoch 4 --data_type 'clusterless' --n_cores 1 --n_workers 1 --wall_time '00:30:00'
python queue_cluster_jobs.py --Animal 'remy' --Day 37 --Epoch 2 --data_type 'clusterless' --n_cores 1 --n_workers 1 --wall_time '00:30:00'
python queue_cluster_jobs.py --Animal 'remy' --Day 37 --Epoch 4 --data_type 'clusterless' --n_cores 1 --n_workers 1 --wall_time '00:30:00'
