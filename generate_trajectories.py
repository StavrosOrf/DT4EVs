import os
import time
import numpy as np
import pickle
import yaml
from tqdm import tqdm
import shutil
import gzip

from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.utilities.arg_parser import arg_parser
from ev2gym.rl_agent.reward import SquaredTrackingErrorReward, ProfitMax_TrPenalty_UserIncentives, profit_maximization, SimpleReward
from ev2gym.rl_agent.state import V2G_profit_max, PublicPST, V2G_profit_max_loads
from ev2gym.baselines.heuristics import RandomAgent, RoundRobin_GF, ChargeAsFastAsPossible
from ev2gym.baselines.gurobi_models.PST_V2G_profit_max_mo import mo_PST_V2GProfitMaxOracleGB
from utils import PST_V2G_ProfitMax_reward, PST_V2G_ProfitMaxGNN_state, PST_V2G_ProfitMax_state

from ev2gym.baselines.mpc.eMPC_v2 import eMPC_V2G_v2


if __name__ == "__main__":

    args = arg_parser()
    SAVE_EVAL_REPLAYS = args.save_eval_replays

    args.config_file = "./config_files/PST_V2G_ProfixMax_50_PES.yaml"

    reward_function = PST_V2G_ProfitMax_reward
    state_function = PST_V2G_ProfitMax_state
    problem = args.config_file.split("/")[-1].split(".")[0]

    env = EV2Gym(config_file=args.config_file,
                 state_function=state_function,
                 reward_function=reward_function,
                 save_replay=SAVE_EVAL_REPLAYS,
                 )

    temp_env = EV2Gym(config_file=args.config_file,
                      save_replay=True,
                      reward_function=reward_function,
                      state_function=state_function,
                      )

    n_trajectories = args.n_trajectories

    config = yaml.load(open(args.config_file, 'r'), Loader=yaml.FullLoader)
    number_of_charging_stations = config["number_of_charging_stations"]
    n_transformers = config["number_of_transformers"]
    steps = config["simulation_length"]
    timescale = config["timescale"]

    trajectories = []
    if args.dataset not in ["random", "optimal", "bau",
                            "mixed_bau_50", "mixed_bau_25", "mixed_bau_75"
                            ]:
        raise ValueError(
            f"Trajectories type {args.dataset} not supported")

    trajecotries_type = args.dataset  # random, optimal, bau

    file_name = f"{problem}_{trajecotries_type}_{number_of_charging_stations}_{n_trajectories}.pkl"
    save_folder_path = f"./trajectories/"
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    # make eval replay folder
    if SAVE_EVAL_REPLAYS:
        if not os.path.exists("eval_replays"):
            os.makedirs("eval_replays")

        file_name = f"{problem}_{trajecotries_type}_{number_of_charging_stations}_{n_trajectories}"
        save_folder_path = f"./eval_replays/" + file_name
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)

        print(f"Saving evaluation replays to {save_folder_path}")

    epoch = 0
    # use tqdm with a fancy bar
    for i in tqdm(range(n_trajectories)):

        trajectory_i = {"observations": [],
                        "actions": [],
                        "rewards": [],
                        "dones": [],
                        "action_mask": [],
                        }

        epoch_return = 0

        if trajecotries_type == "random":
            agent = RandomAgent(env)
        elif trajecotries_type == "bau":
            agent = RoundRobin_GF(env)

        elif trajecotries_type == "mixed_bau_50":
            if i % 2 == 0:
                agent = RoundRobin_GF(env)
            else:
                agent = RandomAgent(env)

        elif trajecotries_type == "mixed_bau_25":
            if i % 4 == 0:
                agent = RoundRobin_GF(env)
            else:
                agent = RandomAgent(env)

        elif trajecotries_type == "mixed_bau_75":
            if i % 4 == 0:
                agent = RandomAgent(env)
            else:
                agent = RoundRobin_GF(env)

        elif trajecotries_type == "optimal":
            _, _ = temp_env.reset()
            agent = ChargeAsFastAsPossible()

            for _ in range(temp_env.simulation_length):
                actions = agent.get_action(temp_env)
                new_state, reward, done, truncated, stats = temp_env.step(
                    actions)  # takes action
                if done:
                    break

            new_replay_path = f"./replay/replay_{temp_env.sim_name}.pkl"

            timelimit = 120

            agent = mo_PST_V2GProfitMaxOracleGB(new_replay_path,
                                                timelimit=timelimit,
                                                MIPGap=None,
                                                )

        elif trajecotries_type == "mpc":
            agent = eMPC_V2G_v2(env,
                                control_horizon=10,
                                MIPGap=0.1,
                                time_limit=30,
                                verbose=False)
        else:
            raise ValueError(
                f"Trajectories type {trajecotries_type} not supported")

        if trajecotries_type == "optimal":
            env = EV2Gym(config_file=args.config_file,
                         load_from_replay_path=new_replay_path,
                         state_function=state_function,
                         reward_function=reward_function,
                         save_replay=SAVE_EVAL_REPLAYS,
                         )
            os.remove(new_replay_path)

        state, _ = env.reset()

        if SAVE_EVAL_REPLAYS:
            env.eval_mode = "optimal"

        while True:

            actions = agent.get_action(env)

            new_state, reward, done, truncated, stats = env.step(actions)

            trajectory_i["observations"].append(state)
            trajectory_i["actions"].append(actions)
            trajectory_i["rewards"].append(reward)
            trajectory_i["dones"].append(done)
            trajectory_i["action_mask"].append(stats['action_mask'])

            state = new_state

            if done:
                # move the replay file to the eval replay folder
                if SAVE_EVAL_REPLAYS:
                    replay_path = env.replay_path + 'replay_' + env.sim_name + '.pkl'
                    new_replay_path = f"./eval_replays/{file_name}/replay_{env.sim_name}_{i}.pkl"
                    shutil.move(replay_path, new_replay_path)

                break
        print(f'Stats: {env.stats["total_reward"]}')
        trajectory_i["observations"] = np.array(trajectory_i["observations"])
        trajectory_i["actions"] = np.array(trajectory_i["actions"])
        trajectory_i["rewards"] = np.array(trajectory_i["rewards"])
        trajectory_i["dones"] = np.array(trajectory_i["dones"])
        trajectory_i["action_mask"] = np.array(trajectory_i["action_mask"])

        trajectories.append(trajectory_i)

        if trajecotries_type == "optimal":
            divident = 100
        else:
            divident = 1000

        if i % divident == 0 and not SAVE_EVAL_REPLAYS and i > 0:
            print(f'Saving trajectories to {save_folder_path+file_name}')

            with gzip.open(save_folder_path+file_name+".gz", 'wb') as f:
                pickle.dump(trajectories, f)

    env.close()

    if SAVE_EVAL_REPLAYS:
        print(
            f'Genereated {n_trajectories} trajectories and saved them in {save_folder_path}')
    else:
        # print(trajectories[:1])
        print(f'Saving trajectories to {save_folder_path+file_name}')

        with gzip.open(save_folder_path+file_name+".gz", 'wb') as f:
            pickle.dump(trajectories, f)

        # To read the compressed pickle file
        with gzip.open(save_folder_path+file_name+".gz", 'rb') as f:
            loaded_data = pickle.load(f)

        print(loaded_data)
