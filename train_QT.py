from ev2gym.models.ev2gym_env import EV2Gym
from utils import PST_V2G_ProfitMax_reward, PST_V2G_ProfitMaxGNN_state, PST_V2G_ProfitMax_state

import numpy as np
import torch

import argparse
import pickle
import random
import sys
import os
import pathlib
import time
import yaml
import wandb

from QT.evaluation.evaluate_episodes import evaluate_episode_rtg
from QT.training.ql_trainer import Trainer
from QT.models.ql_DT import DecisionTransformer, Critic


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    warmup_tokens = 375e6
    final_tokens = 260e9  # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 8  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def save_checkpoint(state, name):
    filename = name
    torch.save(state, filename)


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def experiment(
        exp_prefix,
        vars,
):

    device = torch.device(vars['device'])
    print(f"Device: {device}")
    log_to_wandb = vars.get('log_to_wandb', False)

    env_name, dataset = vars['env'], vars['dataset']
    # model_type = vars['model_type']
    # group_name = f'{exp_prefix}-{env_name}'

    run_name = vars['name']

    exp_prefix = f'{run_name}_{random.randint(int(1e5), int(1e6) - 1)}'

    # seed everything
    seed = vars['seed']

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    scale = 1
    env_targets = 0  # evaluation conditioning targets

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
    
    elif dataset == 'optimal_250_5000':
        dataset_path = 'trajectories/PST_V2G_ProfixMax_250_optimal_250_10000.pkl.gz'
    elif dataset == 'random_250_5000':
        dataset_path = 'trajectories/PST_V2G_ProfixMax_250_random_250_10000.pkl.gz'
    elif dataset == 'bau_250_5000':
        dataset_path = 'trajectories/PST_V2G_ProfixMax_250_bau_250_10000.pkl.gz'
    
    else:
        raise NotImplementedError("Dataset not found")


    max_ep_len = steps
    g_name = vars['group_name']

    group_name = f'{g_name}DT_{number_of_charging_stations}cs'

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

    print(f'Loaded {len(eval_envs)} evaluation replays')

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

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

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
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(
        states, axis=0), np.std(states, axis=0) + 1e-6

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

        s, a, r, d, rtg, timesteps, mask, target_a = [], [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            # get sequences from dataset
            s.append(traj['observations']
                     [si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            target_a.append(
                traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1, 1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1, 1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >=
                          max_ep_len] = max_ep_len-1  # padding cutoff

            if vars['reward_tune'] == 'cql_antmaze':
                traj_rewards = (traj['rewards']-0.5) * 4.0
            else:
                traj_rewards = traj['rewards']
            r.append(traj_rewards[si:si + max_len].reshape(1, -1, 1))
            rtg.append(discount_cumsum(traj_rewards[si:], gamma=1.)[
                       :s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1],
                                         np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len -
                                   tlen, state_dim)), s[-1]], axis=1)
            
            # s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.zeros((1, max_len -
                                   tlen, act_dim)), a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len -
                                   tlen, 1)), r[-1]], axis=1)
            target_a[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen, act_dim)), target_a[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len -
                                   tlen, 1)), d[-1]], axis=1)
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
        target_a = torch.from_numpy(np.concatenate(target_a, axis=0)).to(
            dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(
            dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(
            dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(
            dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, target_a, d, rtg, timesteps, mask

    def eval_episodes(target_rew):
        def fn(model, critic):
            with torch.no_grad():
                stats = evaluate_episode_rtg(
                    eval_envs,
                    state_dim,
                    act_dim,
                    model,
                    critic,
                    max_ep_len=max_ep_len,
                    scale=scale,
                    target_return=target_rew/scale,
                    mode=mode,
                    state_mean=state_mean,
                    state_std=state_std,
                    device=device,
                )
            return stats
        return fn

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
        n_positions=1024,
        resid_pdrop=vars['dropout'],
        attn_pdrop=vars['dropout'],
        scale=scale,
        sar=vars['sar'],
        rtg_no_q=vars['rtg_no_q'],
        infer_no_q=vars['infer_no_q']
    )
    critic = Critic(
        state_dim, act_dim, hidden_dim=vars['embed_dim']
    )

    model = model.to(device=device)
    critic = critic.to(device=device)

    trainer = Trainer(
        model=model,
        critic=critic,
        batch_size=batch_size,
        tau=vars['tau'],
        discount=vars['discount'],
        get_batch=get_batch,
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean(
            (a_hat - a)**2),
        eval_fns=[eval_episodes(env_targets)],
        max_q_backup=vars['max_q_backup'],
        eta=vars['eta'],
        eta2=vars['eta2'],
        ema_decay=0.995,
        step_start_ema=1000,
        update_ema_every=5,
        lr=vars['learning_rate'],
        weight_decay=vars['weight_decay'],
        lr_decay=vars['lr_decay'],
        lr_maxt=vars['max_iters'],
        lr_min=vars['lr_min'],
        grad_norm=vars['grad_norm'],
        scale=scale,
        k_rewards=vars['k_rewards'],
        use_discount=vars['use_discount']
    )

    log_to_wandb = vars.get('log_to_wandb', False)
    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='DT4EVs',
            save_code=True,
            config=vars
        )

    best_reward = - np.Inf

    for iter in range(vars['max_iters']):
        outputs = trainer.train_iteration(num_steps=vars['num_steps_per_iter'],
                                          logger=None,
                                          iter_num=iter+1,
                                          log_writer=None)

        trainer.scale_up_eta(vars['lambda'])

        if outputs['test/total_reward'] > best_reward:
            best_reward = outputs['test/total_reward']
            # save pytorch model
            torch.save(model.state_dict(),
                       f'saved_models/{exp_prefix}/model.best')
            torch.save(critic.state_dict(),
                       f'saved_models/{exp_prefix}/critic.best')
            print(
                f' Saving best model with reward {best_reward} at path saved_models/{exp_prefix}/model.best')

        outputs['best'] = best_reward

        if log_to_wandb:
            wandb.log(outputs)

    torch.save(model.state_dict(), f'{save_path}/model.last')
    torch.save(critic.state_dict(), f'{save_path}/critic.last')

    if log_to_wandb:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PST_V2G_ProfixMax_1000')
    parser.add_argument('--name', type=str, default='QT')
    parser.add_argument('--group_name', type=str, default='2ndTests_')
    parser.add_argument('--seed', type=int, default=42)
    # medium, medium-replay, medium-expert, expert
    parser.add_argument('--dataset', type=str, default='random_100')
    # normal for standard setting, delayed for sparse
    parser.add_argument('--mode', type=str, default='normal')
    parser.add_argument('--K', type=int, default=2)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=3e-4)
    parser.add_argument('--lr_min', type=float, default=0.)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=10)
    parser.add_argument('--max_iters', type=int, default=250)
    parser.add_argument('--num_steps_per_iter', type=int, default=1000)  # 1000
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_path', type=str, default='./save/')

    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--eta", default=0.01, type=float)
    parser.add_argument("--eta2", default=1.0, type=float)
    parser.add_argument("--lambda", default=1.0, type=float)
    parser.add_argument("--max_q_backup", action='store_true', default=False)
    parser.add_argument("--lr_decay", action='store_true', default=True)
    parser.add_argument("--grad_norm", default=2.0, type=float)
    parser.add_argument("--early_stop", action='store_true', default=False)
    parser.add_argument("--early_epoch", type=int, default=100)
    parser.add_argument("--k_rewards", action='store_true', default=True)
    parser.add_argument("--use_discount", action='store_true', default=True)
    parser.add_argument("--sar", action='store_true', default=False)
    parser.add_argument("--reward_tune", default='no', type=str)
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--test_scale", type=float, default=1)
    parser.add_argument("--rtg_no_q", action='store_true', default=False)
    parser.add_argument("--infer_no_q", action='store_true', default=False)

    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    parser.add_argument('--eval_replay_path', type=str,
                        default="./eval_replays/PST_V2G_ProfixMax_25_optimal_25_50/")
    parser.add_argument('--config_file', type=str,
                        default="PST_V2G_ProfixMax_25.yaml")

    args = parser.parse_args()

    experiment(args.name, vars=vars(args))
