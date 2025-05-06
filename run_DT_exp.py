"""
This file is used to run various experiments in different tmux panes each.
"""

import os
import time
import random

# run train_DT.py in a tmux pane for each K and dataset

batch_size = 128
num_steps_per_iter = 1000
max_iters = 250
num_eval_episodes = 30
embed_dim = 128

counter = 0

# DT: dt
# DT + DT + state-GNN: gnn_dt
# DT + state-GNN + residual connection: gnn_in_out_dt
# DT + state-GNN + action-GNN residual connection: --
# DT + state-GNN + action-GNN residual connection + no action masking: gnn_act_embNoMask
# GNN-DT with GAT: 10
# GNN-DT (full version): gnn_act_emb

for model_type in [
                  "bc",  # behavior cloning
                   ]:  # dt, gnn_dt, gnn_in_out_dt, bc, gnn_act_emb
    for action_mask in [False]:
    # for action_mask in [True]:

        # if model_type != "gnn_act_emb" and not action_mask:
        #     continue

        for K in [10]:
            # for gnn_type in ['GCN', 'GAT']:

            #     if gnn_type != 'GCN' and model_type != 'gnn_act_emb':
            #         continue
                
                # if model_type == "gnn_act_emb" and not action_mask and gnn_type != 'GCN':
                #     continue
                            
                for dataset in ["optimal_10000",
                                "bau_10000",
                                "random_10000",
                                ]:
                    for seed in [10]:  # 128, 512
                        for n_layer, n_head in [(3, 4)]:  # (3, 1),(3,4)

                            # a10 machine config
                            # command = 'tmux new-session -d \; send-keys " /home/sorfanouda/anaconda3/envs/dt/bin/python train_DT.py' + \
                            # iepg machine config
                            # command = 'tmux new-session -d \; send-keys "  /home/sorfanoudakis/.conda/envs/dt3/bin/python train_DT.py' + \

                            temp_name = model_type
                            # if model_type == "gnn_act_emb" and not action_mask:
                            #     temp_name += "NoMask"
                            # elif model_type == "gnn_act_emb" and action_mask:
                            #     temp_name += f"_{gnn_type}"

                            run_name = f'{temp_name}_run_{seed}_K={K}_batch={batch_size}_dataset={dataset}_embed_dim={embed_dim}_n_layer={n_layer}_n_head={n_head}'
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
                                ' --group_name ' + '"offline_RL"' + \
                                ' --name ' +  str(run_name) + \
                                '" Enter'

                            # command = 'tmux new-session -d \; send-keys " /home/sorfanouda/anaconda3/envs/dt/bin/python train_DT.py' + \
                            #     ' --dataset ' + dataset + \
                            #     ' --K ' + str(K) + \
                            #     ' --device cuda:0' + \
                            #     ' --seed ' + str(seed) + \
                            #     ' --model_type ' + model_type + \
                            #     ' --embed_dim ' + str(32) + \
                            #     ' --n_layer ' + str(3) + \
                            #     ' --n_head ' + str(1) + \
                            #     ' --max_iters=' + str(10) + \
                            #     ' --batch_size=' + str(16) + \
                            #     ' --num_steps_per_iter=' + str(2) + \
                            #     ' --num_eval_episodes=' + str(2) + \
                            #     ' --action_masking ' + str(action_mask) + \
                            #     ' --group_name ' + '"ablation"' + \
                            #     ' --gnn_type ' + str(gnn_type) + \
                            #     ' --name ' + str(run_name) + \
                            #     '" Enter'
                            os.system(command=command)
                            print(command)
                            # wait for 20 seconds before starting the next experiment
                            time.sleep(3)
                            counter += 1
