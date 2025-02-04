# This script reads the replay files and evaluates the performance.

from DT.load_model import load_DT_model, load_QT_model, load_GNN_act_emb_DecisionTransformer_model
from DT.evaluation.evaluate_episodes import evaluate_episode_rtg_from_replays
from QT.evaluation.evaluate_episodes import QT_evaluate_episode_rtg_from_replays

import gymnasium as gym
# from state_action_eda import AnalysisReplayBuffer
from ev2gym.visuals.evaluator_plot import plot_comparable_EV_SoC_single, plot_prices
from ev2gym.visuals.evaluator_plot import plot_total_power_V2G, plot_actual_power_vs_setpoint
from ev2gym.visuals.evaluator_plot import plot_total_power, plot_comparable_EV_SoC
from ev2gym.rl_agent.state import V2G_profit_max, PublicPST, V2G_profit_max_loads
from ev2gym.rl_agent.reward import profit_maximization, ProfitMax_TrPenalty_UserIncentives
from ev2gym.rl_agent.reward import SquaredTrackingErrorReward, SimpleReward
from ev2gym.baselines.gurobi_models.profit_max import V2GProfitMaxOracleGB
from ev2gym.baselines.gurobi_models.tracking_error import PowerTrackingErrorrMin

from utils import PST_V2G_ProfitMax_state, PST_V2G_ProfitMax_reward
from ev2gym.baselines.heuristics import RoundRobin_GF

from sb3_contrib import TQC, TRPO, ARS, RecurrentPPO
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
from ev2gym.baselines.mpc.V2GProfitMax import V2GProfitMaxOracle, V2GProfitMaxLoadsOracle
from ev2gym.baselines.mpc.eMPC_v2 import eMPC_V2G_v2
from ev2gym.baselines.mpc.eMPC import eMPC_V2G, eMPC_G2V
from ev2gym.baselines.mpc.ocmf_mpc import OCMF_V2G, OCMF_G2V
from ev2gym.baselines.heuristics import ChargeAsFastAsPossibleToDesiredCapacity
from ev2gym.baselines.heuristics import RoundRobin, ChargeAsLateAsPossible, ChargeAsFastAsPossible
from ev2gym.baselines.gurobi_models.PST_V2G_profit_max_mo import mo_PST_V2GProfitMaxOracleGB

from ev2gym.models.ev2gym_env import EV2Gym
import yaml
import os
import pickle
from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import datetime
import time
import random

import warnings

# Suppress all UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

# set seeds
seed = 10 #6
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)


def evaluator():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ############# Simulation Parameters #################
    n_test_cycles = 1
    SAVE_REPLAY_BUFFER = False
    SAVE_EV_PROFILES = False

    # values in [0-1] probability of communication failure
    # p_fail_list = [0, 0.1, 0.25, 0.5]
    p_fail_list = [0]  # values in [0-1] probability of communication failure

    # p_delay_list = [0, 0.1, 0.2, 0.3]
    # values in [0-1] probability of obs-delayed communication
    p_delay_list = [0]

    # config_file = "./config_files/PST_V2G_ProfixMax_25.yaml"    
    # config_file = "./config_files/PST_V2G_ProfixMax_25_G1.yaml"
    # config_file = "./config_files/PST_V2G_ProfixMax_25_G2.yaml"
    # config_file = "./config_files/PST_V2G_ProfixMax_25_G3.yaml"

    #diffrent CS number comparison
    # config_file = "./config_files/PST_V2G_ProfixMax_25_CS5.yaml"
    # config_file = "./config_files/PST_V2G_ProfixMax_25_CS50.yaml"
    # config_file = "./config_files/PST_V2G_ProfixMax_25_CS75.yaml"
    # config_file = "./config_files/PST_V2G_ProfixMax_25_CS100.yaml"            
    
    # config_file = "./config_files/PST_V2G_ProfixMax_250.yaml"
            
    #### PES exps
    config_file = "./config_files/PST_V2G_ProfixMax_50_PES.yaml"

    if "PST_V2G_ProfixMax" in config_file:
        state_function_Normal = PST_V2G_ProfitMax_state
        state_function_GNN = None
        reward_function = PST_V2G_ProfitMax_reward
    else:
        raise ValueError(f'Unknown config file {config_file}')

    # Algorithms to compare:
    # Use algorithm name or the saved RL model path as string
    algorithms = [
        ChargeAsFastAsPossible,
        RoundRobin_GF,
        # "gnn_in_out_dt_run_20_K=10_batch=128_dataset=optimal_2000_embed_dim=128_n_layer=3_n_head=427839.optimal_2000.527996",
        # "gnn_act_emb_run_42_K=2_batch=128_dataset=optimal_2000_embed_dim=128_n_layer=3_n_head=451760.optimal_2000.835025",

        ################## Best models ##################################
        # 'gnn_act_emb_run_40_K=10_dataset=optimal_25_1000_25724_537244',
        # 'dt_run_10_K=10_dataset=bau_10000_94967_118081',
        # 'QT_run_30_K=2_dataset=optimal_1000_87175_301984',
        #################################################################
        
        ################## Best models PES_50 ##################################
        'ppo_run_1_95066',
        'gnn_act_emb_run_10_K=5_dataset=mixed_50_optimal_random_92276_366128',
        #################################################################
        # ChargeAsLateAsPossible,
        # RoundRobin_GF_off_allowed,
        # RoundRobin_GF,
        # RoundRobin,
        # eMPC_V2G_v2,
        mo_PST_V2GProfitMaxOracleGB,
        # eMPC_V2G,
        
        # # V2GProfitMaxLoadsOracle,
        # V2GProfitMaxOracleGB,
        # V2GProfitMaxOracle,
        # PowerTrackingErrorrMin
    ]

    # create a AnalysisReplayBuffer object for each algorithm

    env = EV2Gym(config_file=config_file,
                 generate_rnd_game=True,
                 state_function=state_function_Normal,
                 reward_function=reward_function,
                 )

    if SAVE_REPLAY_BUFFER:
        replay_buffers = {}
        for algorithm in algorithms:

            replay_buffers[algorithm] = AnalysisReplayBuffer(state_dim=env.observation_space.shape[0],
                                                             action_dim=env.action_space.shape[0],
                                                             max_size=int(1e4))

    #####################################################

    config = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)

    number_of_charging_stations = config["number_of_charging_stations"]
    n_transformers = config["number_of_transformers"]
    timescale = config["timescale"]
    simulation_length = config["simulation_length"]

    scenario = config_file.split("/")[-1].split(".")[0]
    eval_replay_path = f'./replay/{number_of_charging_stations}cs_{n_transformers}tr_{scenario}/'
    print(f'Looking for replay files in {eval_replay_path}')
    try:
        eval_replay_files = [f for f in os.listdir(
            eval_replay_path) if os.path.isfile(os.path.join(eval_replay_path, f))]

        print(
            f'Found {len(eval_replay_files)} replay files in {eval_replay_path}')
        if n_test_cycles > len(eval_replay_files):
            n_test_cycles = len(eval_replay_files)

        replay_to_print = 1
        replay_to_print = min(replay_to_print, len(eval_replay_files)-1)
        replays_exist = True

    except:
        n_test_cycles = n_test_cycles
        replays_exist = False

    print(f'Number of test cycles: {n_test_cycles}')

    if SAVE_EV_PROFILES:
        ev_profiles = []

    def generate_replay(evaluation_name):
        env = EV2Gym(config_file=config_file,
                     generate_rnd_game=True,
                     save_replay=True,
                     replay_save_path=f"replay/{evaluation_name}/",
                     )
        replay_path = f"replay/{evaluation_name}/replay_{env.sim_name}.pkl"

        for _ in range(env.simulation_length):
            actions = np.ones(env.cs)

            new_state, reward, done, truncated, _ = env.step(
                actions, visualize=False)  # takes action

            if done:
                break

        if SAVE_EV_PROFILES:
            ev_profiles.append(env.EVs_profiles)
        return replay_path

    evaluation_name = f'eval_{number_of_charging_stations}cs_{n_transformers}tr_{scenario}_{len(algorithms)}_algos' +\
        f'_{n_test_cycles}_exp_' +\
        f'{datetime.datetime.now().strftime("%Y_%m_%d_%f")}'

    # make a directory for the evaluation
    save_path = f'./results/{evaluation_name}/'
    os.makedirs(save_path, exist_ok=True)
    os.system(f'cp {config_file} {save_path}')

    if not replays_exist:
        eval_replay_files = [generate_replay(
            evaluation_name) for _ in range(n_test_cycles)]

    # save the list of EV profiles to a pickle file
    if SAVE_EV_PROFILES:
        with open(save_path + 'ev_profiles.pkl', 'wb') as f:
            print(f'Saving EV profiles to {save_path}ev_profiles.pkl')
            pickle.dump(ev_profiles, f)

        exit()

    plot_results_dict = {}
    counter = 0

    for p_delay in p_delay_list:
        print(f' +------- Evaluating with p_delay={p_delay} -------+')
        for p_fail in p_fail_list:
            p_delay = p_fail
            print(f' +------- Evaluating with p_fail={p_fail} -------+')
            for algorithm in algorithms:
                print(' +------- Evaluating', algorithm, " -------+")
                for k in range(n_test_cycles):
                    print(f' Test cycle {k+1}/{n_test_cycles} -- {algorithm}')
                    counter += 1
                    h = -1

                    if replays_exist:
                        replay_path = eval_replay_path + eval_replay_files[k]
                    else:
                        replay_path = eval_replay_files[k]

                    if type(algorithm) == str:
                        if "GNN" in algorithm:
                            state_function = state_function_GNN
                        else:
                            state_function = state_function_Normal
                    else:
                        state_function = state_function_Normal

                    env = EV2Gym(config_file=config_file,
                                 load_from_replay_path=replay_path,
                                 state_function=state_function,
                                 reward_function=reward_function,
                                 )

                    # initialize the timer
                    timer = time.time()
                    state, _ = env.reset()
                    try:
                        if type(algorithm) == str:
                            if algorithm.split('_')[0] in ['OCMF', 'eMPC']:
                                h = int(algorithm.split('_')[2])
                                algorithm = algorithm.split(
                                    '_')[0] + '_' + algorithm.split('_')[1]
                                print(
                                    f'Algorithm: {algorithm} with control horizon {h}')
                                if algorithm == 'OCMF_V2G':
                                    model = OCMF_V2G(
                                        env=env, control_horizon=h)
                                    algorithm = OCMF_V2G
                                elif algorithm == 'OCMF_G2V':
                                    model = OCMF_G2V(
                                        env=env, control_horizon=h)
                                    algorithm = OCMF_G2V
                                elif algorithm == 'eMPC_V2G':
                                    model = eMPC_V2G(
                                        env=env, control_horizon=h)
                                    algorithm = eMPC_V2G
                                elif algorithm == 'eMPC_G2V':
                                    model = eMPC_G2V(
                                        env=env, control_horizon=h)
                                    algorithm = eMPC_G2V

                                algorithm_name = algorithm.__name__

                            elif any(algo in algorithm for algo in ['ppo', 'a2c', 'ddpg', 'tqc', 'trpo', 'ars', 'rppo']):

                                gym.envs.register(id='evs-v0', entry_point='ev2gym.models.ev2gym_env:EV2Gym',
                                                  kwargs={'config_file': config_file,
                                                          'generate_rnd_game': True,
                                                          'state_function': state_function_Normal,
                                                          'reward_function': reward_function,
                                                          'load_from_replay_path': replay_path,
                                                          })
                                env = gym.make('evs-v0')

                                load_path = f'./saved_models/{algorithm}/best_model.zip'

                                # initialize the timer
                                timer = time.time()
                                algorithm_name = algorithm.split('_')[0]

                                if 'rppo' in algorithm:
                                    sb3_algo = RecurrentPPO
                                elif 'ppo' in algorithm:
                                    sb3_algo = PPO
                                elif 'a2c' in algorithm:
                                    sb3_algo = A2C
                                elif 'ddpg' in algorithm:
                                    sb3_algo = DDPG
                                elif 'tqc' in algorithm:
                                    sb3_algo = TQC
                                elif 'trpo' in algorithm:
                                    sb3_algo = TRPO
                                else:
                                    exit()

                                model = sb3_algo.load(load_path,
                                                      env,
                                                      device=device)
                                # set replay buffer to None

                                if 'tqc' in algorithm or 'ddpg' in algorithm:
                                    model.replay_buffer = model.replay_buffer.__class__(1,
                                                                                        model.observation_space,
                                                                                        model.action_space,
                                                                                        device=model.device,
                                                                                        optimize_memory_usage=model.replay_buffer.optimize_memory_usage)

                                env = model.get_env()
                                state = env.reset()

                            elif "gnn_act_emb" in algorithm:
                                model_path = algorithm

                                model = load_GNN_act_emb_DecisionTransformer_model(model_path=model_path,
                                                                                   max_ep_len=simulation_length,
                                                                                   env=env,
                                                                                   config=config,
                                                                                   device=device)

                                algorithm_name = "GNN_act_emb_DT"
                                model.eval()

                            elif "dt" in algorithm:
                                model_path = algorithm

                                model, state_mean, state_std = load_DT_model(model_path=model_path,
                                                                             max_ep_len=simulation_length,
                                                                             env=env,
                                                                             #   config=config,
                                                                             device=device)

                                algorithm_name = "DT"
                                model.eval()

                            elif "QT" in algorithm:
                                model_path = algorithm

                                model, critic = load_QT_model(model_path=model_path,
                                                      max_ep_len=simulation_length,
                                                      env=env,
                                                      #   config=config,
                                                      device=device)

                                algorithm_name = "QT"
                                model.eval()

                            else:
                                raise ValueError(
                                    f'Unknown algorithm {algorithm}')

                        else:
                            model = algorithm(env=env,
                                              replay_path=replay_path,
                                              verbose=False,
                                              )
                            algorithm_name = algorithm.__name__

                    except Exception as error:
                        # print(error)
                        # print(
                        #     f'!!!!!!!!!! Error in {algorithm} with replay {replay_path}')
                        # continue
                        raise error

                    rewards = []

                    DT_FLAG = False
                    if type(algorithm) == str:
                        if "dt" in algorithm or "gnn_act_emb" in algorithm or "QT" in algorithm:
                            DT_FLAG = True

                    if DT_FLAG:
                        if "DT" in algorithm_name or "gnn_act_emb" in algorithm_name:
                            
                            result_tuple = evaluate_episode_rtg_from_replays(env=env,
                                                                            model=model,
                                                                            max_ep_len=simulation_length,
                                                                            device='cuda',
                                                                            target_return=0,
                                                                            mode='normal',
                                                                            )
                        elif "QT" in algorithm_name:
                            result_tuple = QT_evaluate_episode_rtg_from_replays(env=env,
                                                                               model=model,
                                                                               critic=critic,
                                                                               max_ep_len=simulation_length,
                                                                               scale=1.,
                                                                               device='cuda',
                                                                               target_return=0,
                                                                               mode='normal',
                                                                               )    
                        
                        stats, rewards = result_tuple[0], [result_tuple[1]]
                        done = True

                    else:
                        for i in range(simulation_length):
                            # print(
                            #     f' Step {i+1}/{simulation_length} -- {algorithm}')
                            ################ Evaluation #############################

                            if type(algorithm) == str:
                                if any(algo in algorithm for algo in ['ppo', 'a2c', 'ddpg', 'tqc', 'trpo', 'ars', 'rppo']):
                                    action, _ = model.predict(
                                        state, deterministic=True)
                                    obs, reward, done, stats = env.step(
                                        action)

                                    if i == simulation_length - 2:
                                        saved_env = deepcopy(
                                            env.get_attr('env')[0])

                                    stats = stats[0]
                                algorithm_name = algorithm.split('_')[0]

                            else:
                                simple_state = state_function_Normal(
                                    env=env)
                                # gnn_state = state_function_GNN(env=env)
                                # ev_indexes = gnn_state['action_mapper']

                                action = model.get_action(env=env)
                                new_state, reward, done, _, stats = env.step(
                                    action)

                                if SAVE_REPLAY_BUFFER:
                                    next_simple_state = state_function_Normal(
                                        env=env)
                                    next_gnn_state = state_function_GNN(
                                        env=env)
                                    replay_buffers[algorithm].add(state=simple_state,
                                                                  action=action,
                                                                  ev_action=action[ev_indexes],
                                                                  next_state=next_simple_state,
                                                                  reward=reward,
                                                                  done=done,
                                                                  gnn_state=gnn_state,
                                                                  gnn_next_state=next_gnn_state)
                            ############################################################

                            rewards.append(reward)

                    if done:
                        results_i = pd.DataFrame({'run': k,
                                                  'Algorithm': algorithm_name,
                                                  'algorithm_version': algorithm,
                                                  'control_horizon': h,
                                                  'p_fail': p_fail,
                                                  'p_delay': p_delay,
                                                  'discharge_price_factor': config['discharge_price_factor'],
                                                  'total_ev_served': stats['total_ev_served'],
                                                  'total_profits': stats['total_profits'],
                                                  'total_energy_charged': stats['total_energy_charged'],
                                                  'total_energy_discharged': stats['total_energy_discharged'],
                                                  'average_user_satisfaction': stats['average_user_satisfaction'],
                                                  'power_tracker_violation': stats['power_tracker_violation'],
                                                  'tracking_error': stats['tracking_error'],
                                                  'energy_tracking_error': stats['energy_tracking_error'],
                                                  'energy_user_satisfaction': stats['energy_user_satisfaction'],
                                                  'min_energy_user_satisfaction': stats['min_energy_user_satisfaction'],
                                                  'std_energy_user_satisfaction': stats['std_energy_user_satisfaction'],
                                                  'total_transformer_overload': stats['total_transformer_overload'],
                                                  'battery_degradation': stats['battery_degradation'],
                                                  'battery_degradation_calendar': stats['battery_degradation_calendar'],
                                                  'battery_degradation_cycling': stats['battery_degradation_cycling'],
                                                  'total_reward': sum(rewards),
                                                  'time': time.time() - timer,
                                                  }, index=[counter])

                        # change name of key to algorithm_name
                        if SAVE_REPLAY_BUFFER:
                            if k == n_test_cycles - 1:
                                replay_buffers[algorithm_name] = replay_buffers.pop(
                                    algorithm)

                        if counter == 1:
                            results = results_i
                        else:
                            results = pd.concat(
                                [results, results_i])

                        if any(algo in algorithm_name for algo in ['ppo', 'a2c', 'ddpg', 'tqc', 'trpo', 'ars', 'rppo']):
                            env = saved_env

                        plot_results_dict[str(
                            algorithm)] = deepcopy(env)

    # save the replay buffers to a pickle file
    if SAVE_REPLAY_BUFFER:
        with open(save_path + 'replay_buffers.pkl', 'wb') as f:
            pickle.dump(replay_buffers, f)

    # save the plot_results_dict to a pickle file
    with open(save_path + 'plot_results_dict.pkl', 'wb') as f:
        pickle.dump(plot_results_dict, f)

        # replace some algorithm_version to other names:
    # change from PowerTrackingErrorrMin -> PowerTrackingError

    # print unique algorithm versions

    results['algorithm_version'] = results['algorithm_version'].astype(str).replace(
        "<class 'ev2gym.baselines.heuristics.ChargeAsFastAsPossible'>", 'ChargeAsFastAsPossible')
    results['algorithm_version'] = results['algorithm_version'].astype(str).replace(
        "<class 'ev2gym.baselines.heuristics.RoundRobin_GF_off_allowed'>", 'RoundRobin_GF_off_allowed')
    results['algorithm_version'] = results['algorithm_version'].astype(str).replace(
        "<class 'ev2gym.baselines.heuristics.RoundRobin_GF'>", 'RoundRobin_GF')
    results['algorithm_version'] = results['algorithm_version'].astype(str).replace(
        "<class 'ev2gym.baselines.heuristics.RoundRobin'>", 'RoundRobin')
    results['algorithm_version'] = results['algorithm_version'].astype(str).replace(
        "<class 'ev2gym.baselines.gurobi_models.tracking_error.PowerTrackingErrorrMin'>",
        'Oracle'
    )
    print(results['algorithm_version'].unique())

    # save the results to a csv file
    results.to_csv(save_path + 'data.csv')

    # drop_columns = ['algorithm_version']
    drop_columns = ['Algorithm']

    results = results.drop(columns=drop_columns)

    results_grouped = results.groupby(
        'algorithm_version',).agg(['mean', 'std'])

    # print columns of the results
    # print(results_grouped.columns)

    # savethe latex results in a txt file
    # with open(save_path + 'results_grouped.txt', 'w') as f:
    #     f.write(results_grouped.to_latex())

    # results_grouped.to_csv('results_grouped.csv')
    # print(results_grouped[['tracking_error', 'energy_tracking_error']])
    # print(results_grouped[['tracking_error',
    #       'total_transformer_overload', 'time']])

    # sort results by tracking error
    results_grouped = results_grouped.sort_values(
        by=('total_reward', 'mean'), ascending=True)

    print(results_grouped[['total_reward',
                           'average_user_satisfaction',
                           'total_profits',
                           'power_tracker_violation']])

    #    ]])
    #    'average_user_satisfaction']])
    # input('Press Enter to continue')

    algorithm_names = []
    for algorithm in algorithms:
        # if class has attribute .name, use it
        if hasattr(algorithm, 'algo_name'):
            algorithm_names.append(algorithm.algo_name)
        elif type(algorithm) == str:
            if "gnn_in_out_dt" in algorithm:
                algorithm_names.append('GNN_Dec_DT')
            elif "gnn_act_emb" in algorithm:
                algorithm_names.append('GNN_act_emb_DT')
                # algorithm_names.append(algorithm.split(
                #     '_')[0] + '_' + algorithm.split('_')[1])

            else:
                algorithm_names.append(algorithm.split('_')[0])
        else:
            algorithm_names.append(algorithm.__name__)

    print(f'Algorithm names: {algorithm_names}')

    print(f'Plottting results at {save_path}')

    plot_total_power(results_path=save_path + 'plot_results_dict.pkl',
                     save_path=save_path,
                     algorithm_names=algorithm_names)

    plot_comparable_EV_SoC(results_path=save_path + 'plot_results_dict.pkl',
                           save_path=save_path,
                           algorithm_names=algorithm_names)

    plot_actual_power_vs_setpoint(results_path=save_path + 'plot_results_dict.pkl',
                                  save_path=save_path,
                                  algorithm_names=algorithm_names)

    plot_total_power_V2G(results_path=save_path + 'plot_results_dict.pkl',
                         save_path=save_path,
                         algorithm_names=algorithm_names)

    plot_comparable_EV_SoC_single(results_path=save_path + 'plot_results_dict.pkl',
                                  save_path=save_path,
                                  algorithm_names=algorithm_names)

    plot_prices(results_path=save_path + 'plot_results_dict.pkl',
                save_path=save_path,
                algorithm_names=algorithm_names)


if __name__ == "__main__":
    evaluator()
