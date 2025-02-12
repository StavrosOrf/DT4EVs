"""
This file is used to run various experiments in different tmux panes each.
"""

import os
import time
import random

# run train_DT.py in a tmux pane for each K and dataset

# batch_size = 64
num_steps_per_iter = 1000
max_iters = 350
num_eval_episodes = 30
seed = 42

counter = 0
for model_type in ["gnn_act_emb"]: #dt, gnn_dt, gnn_in_out_dt, bc, gnn_act_emb
    for action_mask in [True]:
        for K in [2]:
            for batch_size in [128]:
                # "bau_25_1000", "bau_50_1000", "bau_75_1000"
                for dataset in ["bau_250_3000"]:
                    for embed_dim in [128]:  # 128, 512
                        #   ' --device cuda:0' + str(counter % 2) + \
                        for n_layer, n_head in [(3, 4)]:  # (3, 1),(3,4)

                            # a10 machine config
                            # command = 'tmux new-session -d \; send-keys " /home/sorfanouda/anaconda3/envs/dt/bin/python train_DT.py' + \
                            # iepg machine config
                            # command = 'tmux new-session -d \; send-keys "  /home/sorfanoudakis/.conda/envs/dt3/bin/python train_DT.py' + \
                            run_name = f'{model_type}_run_{seed}_K={K}_batch={batch_size}_dataset={dataset}_embed_dim={embed_dim}_n_layer={n_layer}_n_head={n_head}'
                            run_name += str(random.randint(0, 100000))
                            
                            command = 'tmux new-session -d \; send-keys " /home/sorfanouda/anaconda3/envs/dt/bin/python train_DT.py' + \
                                ' --dataset ' + dataset + \
                                ' --K ' + str(K) + \
                                ' --device cuda:0' + \
                                ' --seed ' + str(seed) + \
                                ' --model_type ' + model_type + \
                                ' --embed_dim ' + str(embed_dim) + \
                                ' --n_layer ' + str(n_layer) + \
                                ' --n_head ' + str(n_head) + \
                                ' --max_iters=' + str(max_iters) + \
                                ' --batch_size=' + str(batch_size) + \
                                ' --num_steps_per_iter=' + str(num_steps_per_iter) + \
                                ' --num_eval_episodes=' + str(num_eval_episodes) + \
                                ' --log_to_wandb True' + \
                                ' --action_masking ' + str(action_mask) + \
                                ' --group_name ' + '"BAU_tests_"' + \
                                ' --name ' +  str(run_name) + \
                                '" Enter'
                            os.system(command=command)
                            print(command)
                            # wait for 20 seconds before starting the next experiment
                            time.sleep(15)
                            counter += 1
