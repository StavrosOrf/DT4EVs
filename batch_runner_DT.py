'''
This script is used to run the batch of training sciprts for every algorithms evaluated
#old command# srun --mpi=pmix --job-name=interactive-gpu --partition=gpu --gres=gpu:1 --qos=normal --time=01:00:00 --mem-per-cpu=4096 --pty /bin/bash -il
srun --mpi=pmix --job-name=interactive-gpu --partition=gpu --gpus-per-task=1 --qos=normal --time=01:00:00 --mem-per-cpu=4G --cpus-per-task=1 --ntasks=1 --pty /bin/bash -il
srun --mpi=pmix --job-name=interactive --partition=compute --cpus-per-task=1 --qos=normal --time=01:00:00 --mem-per-cpu=4G--ntasks=1 --pty /bin/bash -il
'''
import os
import random

seeds = [10,20,30,40]
# config = "PST_V2G_ProfixMax_25.yaml"
# eval_replay_path = "./eval_replays/PST_V2G_ProfixMax_25_optimal_25_50/"

# config = "PST_V2G_ProfixMax_250.yaml"
# eval_replay_path = "./eval_replays/PST_V2G_ProfixMax_250_optimal_250_50/"

config = "PST_V2G_ProfixMax_50_PES.yaml"
eval_replay_path = "./eval_replays/PST_V2G_ProfixMax_50_PES_optimal_50_50/"

datasets_list = [
    'random_100',
    'random_1000',
    'random_10000',
    'optimal_100',
    'optimal_1000',
    'optimal_10000',
    'bau_100',
    'bau_1000',
    'bau_10000',
]

mid_datasets_list = [
    'random_1000',
    'optimal_1000',
    'bau_1000',
]

mixed_datasets_list = [
    'bau_25_1000',
    'bau_50_1000',
    'bau_75_1000',
    'optimal_25_1000',
    'optimal_50_1000',
    'optimal_75_1000',
]

mixed_opt_datasets_list = [
    'optimal_25_1000',
    'optimal_50_1000',
    'optimal_75_1000',
]

big_datasets_list = [
    # 'random_250_3000',
    # 'optimal_250_3000',
    'bau_250_3000',
]

datasets_50_list = [
    'random_50_5000',
    # 'optimal_50_5000',
    'bau_50_5000',
]

# Extra arguments for the python script
num_steps_per_iter = 1000

# if directory does not exist, create it
if not os.path.exists('./slurm_logs'):
    os.makedirs('./slurm_logs')

# 'gnn_dt', 'gnn_in_out_dt', 'dt'
for model_type in ['dt','gnn_act_emb']:  # 'dt','gnn_act_emb
    for action_mask in [True]:
        for K in [5]:
            for _ in [128]:
                for dataset in datasets_50_list:
                    for _ in [128]:  # 128, 512
                        for n_layer, n_head in [(3, 4)]:  # (3, 1),(3,4)
                            for counter, seed in enumerate(seeds):

                                # if "250" in config:
                                #     memory = 40
                                #     if model_type == 'gnn_act_emb':
                                #         time = 46
                                #     else:
                                #         time = 20

                                #     cpu_cores = 2

                                #     feature_dim = 16
                                #     GNN_hidden_dim = 64
                                #     num_gcn_layers = 3
                                #     act_GNN_hidden_dim = 64
                                #     max_iters = 400
                                #     batch_size = 64
                                #     num_steps_per_iter = 3000
                                #     embed_dim = 256

                                # elif "25" in config:


                                memory = 25

                                if model_type == 'gnn_act_emb':
                                    time = 48
                                else:
                                    time = 25


                                if time > 46:
                                    time = 46

                                cpu_cores = 1

                                feature_dim = 8
                                GNN_hidden_dim = 32
                                num_gcn_layers = 3
                                act_GNN_hidden_dim = 32
                                max_iters = 250
                                batch_size = 128
                                num_steps_per_iter = 1000
                                embed_dim = 128


                                # run_name = f'{algorithm}_run_{counter}_{random.randint(0, 100000)}'
                                run_name = f'{model_type}_run_{seed}_K={K}_dataset={dataset}_'
                                run_name += str(random.randint(0, 100000))
                                # gpu-a100, gpu
                                command = '''#!/bin/sh
#!/bin/bash
#SBATCH --job-name="dt_pes"
#SBATCH --partition=gpu
''' + \
                                    f'#SBATCH --time={time}:00:00' + \
                                    '''
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
''' + \
                                    f'#SBATCH --cpus-per-task={cpu_cores}' + \
                                    '''
''' + \
                                    f'#SBATCH --mem-per-cpu={memory}G' + \
                                    '''
#SBATCH --account=research-eemcs-ese

''' + \
                                    f'#SBATCH --output=./slurm_logs/{run_name}.out' + \
                                    '''
''' + \
                                    f'#SBATCH --error=./slurm_logs/{run_name}.err' + \
                                    '''

module load 2024r1 openmpi miniconda3 py-pip

# Set conda env:
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate dt3
previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

''' + 'srun python train_DT.py' + \
                                    ' --dataset ' + dataset + \
                                    ' --K ' + str(K) + \
                                    ' --device cuda:0' + \
                                    ' --model_type ' + model_type + \
                                    ' --embed_dim ' + str(embed_dim) + \
                                    ' --n_layer ' + str(n_layer) + \
                                    ' --n_head ' + str(n_head) + \
                                    ' --seed ' + str(seed) + \
                                    ' --max_iters=' + str(max_iters) + \
                                    ' --batch_size=' + str(batch_size) + \
                                    ' --num_steps_per_iter=' + str(num_steps_per_iter) + \
                                    ' --log_to_wandb True' + \
                                    ' --feature_dim ' + str(feature_dim) + \
                                    ' --GNN_hidden_dim ' + str(GNN_hidden_dim) + \
                                    ' --act_GNN_hidden_dim ' + str(act_GNN_hidden_dim) + \
                                    ' --action_masking ' + str(action_mask) + \
                                    ' --group_name ' + '"PaperExps"' + \
                                    ' --config_file ' + config + \
                                    ' --eval_replay_path ' + eval_replay_path + \
                                    ' --name ' + str(run_name) + \
                                    '' + \
                                    '''
            
/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"

conda deactivate
'''

                                with open(f'run_tmp.sh', 'w') as f:
                                    f.write(command)

                                with open(f'./slurm_logs/{run_name}.sh', 'w') as f:
                                    f.write(command)

                                os.system('sbatch run_tmp.sh')

                                # command = 'tmux new-session -d \; send-keys " /home/sorfanouda/anaconda3/envs/dt/bin/python train_DT.py' + \
                                #     ' --dataset ' + dataset + \
                                #     ' --K ' + str(K) + \
                                #     ' --device cuda:0' + \
                                #     ' --model_type ' + model_type + \
                                #     ' --embed_dim ' + str(embed_dim) + \
                                #     ' --n_layer ' + str(n_layer) + \
                                #     ' --n_head ' + str(n_head) + \
                                #     ' --seed ' + str(seed) + \
                                #     ' --max_iters=' + str(1) + \
                                #     ' --batch_size=' + str(batch_size) + \
                                #     ' --num_steps_per_iter=' + str(1) + \
                                #     ' --feature_dim ' + str(feature_dim) + \
                                #     ' --GNN_hidden_dim ' + str(GNN_hidden_dim) + \
                                #     ' --act_GNN_hidden_dim ' + str(act_GNN_hidden_dim) + \
                                #     ' --action_masking ' + str(action_mask) + \
                                #     ' --group_name ' + '"DT_Tests_"' + \
                                #     ' --config_file ' + config + \
                                #     ' --eval_replay_path ' + eval_replay_path + \
                                #     ' --name ' + str(run_name) + \
                                #     '" Enter'
                                    
                                # os.system(command=command)
                                # print(command)       
                                # import time as timer                      
                                # timer.sleep(5)
