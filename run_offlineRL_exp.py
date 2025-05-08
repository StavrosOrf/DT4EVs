"""
This file is used to run various experiments in different tmux panes each.
"""

import os
import time

seeds = [20]
epochs = 150
steps_per_epoch = 1000 # 200_000

for dataset in ['bau','optimal','random']:
    for algorithm in ['bc','iql', 'cql']:
    # for algorithm in ['bc']:
        for seed in seeds:
            
            dataset_path = './trajectories/PST_V2G_ProfixMax_25_' + dataset + '_25_10000.pkl.gz'
            exp_name = f'{algorithm}_{dataset}_seed_{seed}'
            
            command = 'tmux new-session -d \; send-keys "/home/sorfanouda/anaconda3/envs/dt/bin/python train_offline_RL_baselines.py' + \
                ' --algo ' + algorithm + \
                ' --device cuda:0' + \
                ' --n_steps ' + str(epochs * steps_per_epoch) + \
                ' --dataset_path ' + dataset_path + \
                ' --exp_name ' + exp_name + \
                ' --seed ' + str(seed) + \
                '" Enter'
            os.system(command=command)
            print(command)
            # wait for 10 seconds before starting the next experiment
            time.sleep(2)
            # counter += 1