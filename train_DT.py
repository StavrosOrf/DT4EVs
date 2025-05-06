# import gymnasium as gym
import numpy as np
import torch
import wandb

import argparse
import pickle
import random
import sys
import os
import yaml

from DT.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from DT.models.decision_transformer import DecisionTransformer
from DT.models.mlp_bc import MLPBCModel
from DT.training.act_trainer import ActTrainer
from DT.training.seq_trainer import SequenceTrainer

from DT.models.gnn_decision_transformer import GNN_DecisionTransformer
from DT.models.gnn_In_Out_decision_transformer import GNN_IN_OUT_DecisionTransformer
from DT.models.gnn_emb_decision_transformer import GNN_act_emb_DecisionTransformer

from ev2gym.models.ev2gym_env import EV2Gym
from utils import PST_V2G_ProfitMax_reward, PST_V2G_ProfitMaxGNN_state, PST_V2G_ProfitMax_state


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def experiment(vars):
    
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = torch.device(vars['device'])
    print(f"Device: {device}")
    log_to_wandb = vars.get('log_to_wandb', False)

    env_name, dataset = vars['env'], vars['dataset']
    model_type = vars['model_type']
    # group_name = f'{exp_prefix}-{env_name}'

    run_name = vars['name']

    exp_prefix = f'{run_name}_{random.randint(int(1e5), int(1e6) - 1)}'

    # seed everything
    seed = vars['seed']

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    scale = 1
    env_targets = [0]  # evaluation conditioning targets

    if model_type == 'bc':
        # since BC ignores target, no need for different evaluations
        env_targets = env_targets[:1]

    config_path = f'./config_files/{vars["config_file"]}'
    config = yaml.load(open(config_path, 'r'),
                       Loader=yaml.FullLoader)

    number_of_charging_stations = config["number_of_charging_stations"]
    steps = config["simulation_length"]

    reward_function = PST_V2G_ProfitMax_reward
    state_function = PST_V2G_ProfitMax_state

    env = EV2Gym(config_file=config_path,
                 state_function=state_function,
                 reward_function=reward_function,
                 )

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print(
        f'Observation space: {env.observation_space.shape[0]}, action space: {env.action_space.shape[0]}')

    # load dataset
    if dataset == 'random_100':
        dataset_path = 'trajectories/PST_V2G_ProfixMax_25_random_25_100.pkl.gz'
    elif dataset == 'random_1000':
        dataset_path = 'trajectories/PST_V2G_ProfixMax_25_random_25_1000.pkl.gz'
    elif dataset == 'random_10000':
        dataset_path = 'trajectories/PST_V2G_ProfixMax_25_random_25_10000.pkl.gz'
        
    elif dataset == 'optimal_100':
        dataset_path = 'trajectories/PST_V2G_ProfixMax_25_optimal_25_100.pkl.gz'
    elif dataset == 'optimal_1000':
        dataset_path = 'trajectories/PST_V2G_ProfixMax_25_optimal_25_1000.pkl.gz'
    elif dataset == 'optimal_10000':
        dataset_path = 'trajectories/PST_V2G_ProfixMax_25_optimal_25_10000.pkl.gz'
        
    elif dataset == 'bau_100':
        dataset_path = 'trajectories/PST_V2G_ProfixMax_25_bau_25_100.pkl.gz'
    elif dataset == 'bau_1000':
        dataset_path = 'trajectories/PST_V2G_ProfixMax_25_bau_25_1000.pkl.gz'
    elif dataset == 'bau_10000':
        dataset_path = 'trajectories/PST_V2G_ProfixMax_25_bau_25_10000.pkl.gz'
        
    elif dataset == 'bau_25_1000':    
        dataset_path = 'trajectories/PST_V2G_ProfixMax_25_mixed_bau_25_25_1000.pkl.gz'
    elif dataset == 'bau_50_1000':
        dataset_path = 'trajectories/PST_V2G_ProfixMax_25_mixed_bau_50_25_1000.pkl.gz'
    elif dataset == 'bau_75_1000':
        dataset_path = 'trajectories/PST_V2G_ProfixMax_25_mixed_bau_75_25_1000.pkl.gz'
        
    elif dataset == 'optimal_25_1000':
        dataset_path = 'trajectories/PST_V2G_ProfixMax_25_mixed_opt_25_25_1000.pkl.gz'
    elif dataset == 'optimal_50_1000':
        dataset_path = 'trajectories/PST_V2G_ProfixMax_25_mixed_opt_50_25_1000.pkl.gz'
    elif dataset == 'optimal_75_1000':
        dataset_path = 'trajectories/PST_V2G_ProfixMax_25_mixed_opt_75_25_1000.pkl.gz'
    
    elif dataset == 'optimal_250_3000':
        dataset_path = 'trajectories/PST_V2G_ProfixMax_250_optimal_250_3000.pkl.gz'
    elif dataset == 'random_250_3000':
        dataset_path = 'trajectories/PST_V2G_ProfixMax_250_random_250_3000.pkl.gz'
    elif dataset == 'bau_250_3000':
        dataset_path = 'trajectories/PST_V2G_ProfixMax_250_bau_250_3000.pkl.gz'
    
    else:
        raise NotImplementedError("Dataset not found")

    
    max_ep_len = steps
    g_name = vars['group_name']

    group_name = f'{g_name}DT_{number_of_charging_stations}cs'

    save_path = f'./saved_models/{exp_prefix}/'
    # create folder
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # save the vars to the save path as yaml
    with open(f'{save_path}/vars.yaml', 'w') as f:
        yaml.dump(vars, f)

    if "gz" in dataset_path:
        import gzip
        with gzip.open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
    else:
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
    
    # save all path information into separate lists
    mode = vars.get('mode', 'normal')
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens = np.array(traj_lens)
    # print(trajectories[0])
    # exit()

    # Initialize eval_envs from replays
    eval_replay_path = vars['eval_replay_path']
    eval_replays = os.listdir(eval_replay_path)
    eval_envs = []
    print(f'Loading evaluation replays from {eval_replay_path}')
    for replay in eval_replays:
        eval_env = EV2Gym(config_file=config_path,
                          load_from_replay_path=eval_replay_path + replay,
                          state_function=state_function,
                          reward_function=reward_function,
                          )
                
        eval_envs.append(eval_env)
        
        if len(eval_envs) >= vars['num_eval_episodes']:
            break

    print(f'Loaded {len(eval_envs)} evaluation replays')

    # used for input normalization
    states = np.concatenate(states, axis=0)

    state_mean, state_std = np.mean(
        states, axis=0), np.std(states, axis=0) + 1e-6

    # save state mean and std
    np.save(f'{save_path}/state_mean.npy', state_mean)
    np.save(f'{save_path}/state_std.npy', state_std)

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(
        f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    K = vars['K']
    batch_size = vars['batch_size']
    num_eval_episodes = vars['num_eval_episodes']
    pct_traj = vars.get('pct_traj', 1.)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask, action_mask = [], [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            # get sequences from dataset
            s.append(traj['observations']
                     [si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            action_mask.append(traj['action_mask']
                               [si:si + max_len].reshape(1, -1, act_dim))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >=
                          max_ep_len] = max_ep_len-1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[
                       :s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1],
                                         np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len -
                                   tlen, state_dim)), s[-1]], axis=1)
            # s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len -
                                   tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len -
                                   tlen, 1)), r[-1]], axis=1)
            action_mask[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen, act_dim)), action_mask[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen))
                                   * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)),
                                     rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate(
                [np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(
            dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(
            dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(
            dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(
            dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(
            dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(
            dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
        action_mask = torch.from_numpy(np.concatenate(action_mask, axis=0)).to(
            dtype=torch.float32, device=device)

        return s, a, r, d, rtg, timesteps, mask, action_mask

    def eval_episodes(target_rew):
        def fn(model):
            with torch.no_grad():
                if model_type == 'dt' or model_type == 'gnn_dt' or model_type == 'gnn_in_out_dt' \
                        or model_type == 'gnn_act_emb':
                    stats = evaluate_episode_rtg(
                        eval_envs,
                        exp_prefix,
                        state_dim,
                        act_dim,
                        model,
                        model_type=model_type,
                        max_ep_len=max_ep_len,
                        scale=scale,
                        target_return=target_rew/scale,
                        mode=mode,
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device,
                        n_test_episodes=num_eval_episodes,
                        config_file=config_path,
                    )
                else:
                    stats = evaluate_episode(
                        eval_envs,
                        state_dim,
                        act_dim,
                        model,
                        max_ep_len=max_ep_len,
                        target_return=target_rew/scale,
                        mode=mode,
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device,
                    )

            return stats

        return fn

    if model_type == 'dt':
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=vars['embed_dim'],
            n_layer=vars['n_layer'],
            n_head=vars['n_head'],
            n_inner=4*vars['embed_dim'],
            activation_function=vars['activation_function'],
            action_masking=vars['action_masking'],
            n_positions=1024,
            resid_pdrop=vars['dropout'],
            attn_pdrop=vars['dropout'],
        )
    elif model_type == 'gnn_dt':
        model = GNN_DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=vars['embed_dim'],
            n_layer=vars['n_layer'],
            n_head=vars['n_head'],
            n_inner=4*vars['embed_dim'],
            activation_function=vars['activation_function'],
            n_positions=1024,
            resid_pdrop=vars['dropout'],
            action_masking=vars['action_masking'],
            attn_pdrop=vars['dropout'],
            action_tanh=True,
            fx_node_sizes={'ev': 5, 'cs': 4, 'tr': 2, 'env': 6},
            feature_dim=vars['feature_dim'],
            GNN_hidden_dim=vars['GNN_hidden_dim'],
            num_gcn_layers=vars['num_gcn_layers'],
            config=config,
            device=device,
        )
    elif model_type == 'gnn_in_out_dt':
        model = GNN_IN_OUT_DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=vars['embed_dim'],
            n_layer=vars['n_layer'],
            n_head=vars['n_head'],
            n_inner=4*vars['embed_dim'],
            activation_function=vars['activation_function'],
            n_positions=1024,
            resid_pdrop=vars['dropout'],
            attn_pdrop=vars['dropout'],
            action_tanh=True,
            action_masking=vars['action_masking'],
            fx_node_sizes={'ev': 5, 'cs': 4, 'tr': 2, 'env': 6},
            feature_dim=vars['feature_dim'],
            GNN_hidden_dim=vars['GNN_hidden_dim'],
            num_gcn_layers=vars['num_gcn_layers'],
            config=config,
            device=device,
        )
    elif model_type == 'gnn_act_emb':
        model = GNN_act_emb_DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=vars['embed_dim'],
            n_layer=vars['n_layer'],
            n_head=vars['n_head'],
            n_inner=4*vars['embed_dim'],
            activation_function=vars['activation_function'],
            n_positions=1024,
            resid_pdrop=vars['dropout'],
            attn_pdrop=vars['dropout'],
            action_tanh=True,
            action_masking=vars['action_masking'],
            fx_node_sizes={'ev': 5, 'cs': 4, 'tr': 2, 'env': 6},
            feature_dim=vars['feature_dim'],
            GNN_hidden_dim=vars['GNN_hidden_dim'],
            num_gcn_layers=vars['num_gcn_layers'],
            act_GNN_hidden_dim=vars['act_GNN_hidden_dim'],
            num_act_gcn_layers=vars['num_act_gcn_layers'],
            config=config,
            device=device,
            gnn_type=vars['gnn_type'],
        )
    elif model_type == 'bc':
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=vars['embed_dim'],
            n_layer=vars['n_layer'],
        )
    else:
        raise NotImplementedError

    model = model.to(device=device)

    warmup_steps = vars['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=vars['learning_rate'],
        weight_decay=vars['weight_decay'],
    )

    max_iters = vars['max_iters']
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda max_iters: min((max_iters+1)/warmup_steps, 1)
    )

    if vars['action_masking']:
        def loss_fn(s_hat, a_hat, r_hat, s, a, r, a_masks): return torch.mean(
            ((a_hat - a)**2) * a_masks
        )

    else:
        def loss_fn(s_hat, a_hat, r_hat, s, a, r, _): return torch.mean(
            (a_hat - a)**2),

    if model_type == 'dt' or model_type == 'gnn_dt' or model_type == 'gnn_in_out_dt' or model_type == 'gnn_act_emb':
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=loss_fn,
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )
    elif model_type == 'bc':
        # raise ValueError
        trainer = ActTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean(
                (a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            entity='stavrosorf',
            project='DT4EVs',
            save_code=True,
            config=vars
        )

    num_steps_per_iter = vars['num_steps_per_iter']
    # num_steps_per_iter = int(1000/batch_size)
    # if num_steps_per_iter == 0:
    #     num_steps_per_iter = vars['batch_size']

    best_reward = -np.Inf

    for iter in range(vars['max_iters']):
        outputs = trainer.train_iteration(
            num_steps=num_steps_per_iter, iter_num=iter+1, print_logs=True)

        if outputs['test/total_reward'] > best_reward:
            best_reward = outputs['test/total_reward']
            # save pytorch model
            torch.save(model.state_dict(),
                       f'saved_models/{exp_prefix}/model.best')
            print(
                f' Saving best model with reward {best_reward} at path saved_models/{exp_prefix}/model.best')

        outputs['best'] = best_reward

        if log_to_wandb:
            wandb.log(outputs)

    torch.save(model.state_dict(), f'{save_path}/model.last')

    if log_to_wandb:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PST_V2G_ProfixMax')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--group_name', type=str, default='tests_')
    parser.add_argument('--seed', type=int, default=42)

    # medium, medium-replay, medium-expert, expert
    parser.add_argument('--dataset', type=str, default='random_100')
    # normal for standard setting, delayed for sparse
    parser.add_argument('--mode', type=str, default='normal')
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=2)
    # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--model_type', type=str,
                        default='bc')  # dt, gnn_dt, gnn_in_out_dt, bc, gnn_act_emb
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--max_iters', type=int, default=500)
    parser.add_argument('--num_steps_per_iter', type=int, default=10)  # 1000
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    parser.add_argument('--config_file', type=str,
                        default="PST_V2G_ProfixMax_25.yaml")

    parser.add_argument('--num_eval_episodes', type=int, default=50)
    parser.add_argument('--eval_replay_path', type=str,
                        default="./eval_replays/PST_V2G_ProfixMax_25_optimal_25_50/")

    # New parameters
    parser.add_argument('--action_masking',
                        type=bool,
                        default=True)

    # GNN_DT parameters
    parser.add_argument('--feature_dim', type=int, default=8)
    parser.add_argument('--GNN_hidden_dim', type=int, default=32)
    parser.add_argument('--num_gcn_layers', type=int, default=3)
    parser.add_argument('--act_GNN_hidden_dim', type=int, default=32)
    parser.add_argument('--num_act_gcn_layers', type=int, default=3)
    parser.add_argument('--gnn_type', type=str, default='GCN')

    args = parser.parse_args()

    experiment(vars=vars(args))
