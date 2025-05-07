import numpy as np
import argparse
import os
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import CQLConfig, IQLConfig, BCConfig
import d3rlpy
from d3rlpy.metrics import EnvironmentEvaluator

from ev2gym.models.ev2gym_env import EV2Gym
from utils import PST_V2G_ProfitMax_reward, PST_V2G_ProfitMaxGNN_state, PST_V2G_ProfitMax_state
import gzip
import pickle

from typing import Any, Dict, Optional

from d3rlpy.logging.logger import LoggerAdapter, LoggerAdapterFactory, SaveProtocol


def load_trajectories(path):
    """
    Load a dataset of trajectories stored as a NumPy .npy file.
    Each trajectory must be a dict with:
    - observations: (T+1, obs_dim)
    - actions: (T, action_dim)
    - rewards: (T,)
    - terminals: (T,)
    """

    with gzip.open(path, 'rb') as f:
        data = pickle.load(f)
    # trajectories = np.load(path, allow_pickle=True)
    print(f"Loaded {len(data)} trajectories from {path}")
    # print(f"First trajectory length: {data[0]}")
    return data


def preprocess_trajectories(trajectories):
    """
    Flatten trajectory data into MDPDataset-compatible format.
    """
    all_obs, all_actions, all_rewards, all_terminals = [], [], [], []

    flag = False
    for traj in trajectories:
        obs = traj["observations"]
        actions = traj["actions"]
        rewards = traj["rewards"]
        terminals = traj["dones"]
        if not flag:
            print(f"áctions type and shape: {actions.shape} {actions.dtype}")
            print(f"observations type and shape: {obs.shape} {obs.dtype}")
            print(f"rewards type and shape: {rewards.shape} {rewards.dtype}")
            print(
                f"terminals type and shape: {terminals.shape} {terminals.dtype}")
            flag = True

        all_obs.append(obs)  # T
        all_actions.append(actions)  # T
        all_rewards.append(rewards)
        all_terminals.append(terminals)

    observations = np.concatenate(all_obs, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    rewards = np.concatenate(all_rewards, axis=0)
    terminals = np.concatenate(all_terminals, axis=0)

    return MDPDataset(observations, actions, rewards, terminals)


def main(args):
    # Load dataset
    print(f"Loading dataset from {args.dataset_path}")
    trajectories = load_trajectories(args.dataset_path)
    dataset = preprocess_trajectories(trajectories)

    # Setup evaluation environment
    config_path = f'./config_files/{args.config_file}'
    # config = yaml.load(open(config_path, 'r'),
    #                    Loader=yaml.FullLoader)

    # number_of_charging_stations = config["number_of_charging_stations"]
    # steps = config["simulation_length"]
    
    # set torch seed
    d3rlpy.seed(args.seed)

    reward_function = PST_V2G_ProfitMax_reward
    state_function = PST_V2G_ProfitMax_state

    eval_env = EV2Gym(config_file=config_path,
                      state_function=state_function,
                      reward_function=reward_function,
                      )

    # Choose algorithm
    if args.algo == "cql":
        config = CQLConfig()
    elif args.algo == "iql":
        config = IQLConfig()
    elif args.algo == "bc":
        config = BCConfig()
    else:
        raise ValueError("Unsupported algorithm. Use 'cql' or 'iql'.")

    algo = config.create(device=args.device)

    wandb_logger = WanDBAdapterFactory(
        project=args.wandb_project,
        # experiment_name=args.wandb_run,
    )

    # Train with automatic WandB logging
    print("Starting training...")
    algo.fit(
        dataset,
        n_steps=args.n_steps,
        evaluators={
            'environment': EnvironmentEvaluator(eval_env,
                                                n_trials=args.num_eval_episodes,
                                                ),
        },
        logger_adapter=wandb_logger,
    )
    print("Training complete.")


class WanDBAdapter(LoggerAdapter):
    r"""WandB Logger Adapter class.

    This class logs data to Weights & Biases (WandB) for experiment tracking.

    Args:
        experiment_name (str): Name of the experiment.
    """

    def __init__(
        self,
        experiment_name: str,
        project: Optional[str] = None,
    ):
        try:
            import wandb
        except ImportError as e:
            raise ImportError("Please install wandb") from e
        self.run = wandb.init(project=project,
                              group="test",
                              name=experiment_name)

        # create save directory
        self.save_dir = f"./saved_models/{experiment_name}"
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Saving models to {self.save_dir}")

    def write_params(self, params: Dict[str, Any]) -> None:
        """Writes hyperparameters to WandB config."""
        self.run.config.update(params)

    def before_write_metric(self, epoch: int, step: int) -> None:
        """Callback executed before writing metric."""

    def write_metric(
        self, epoch: int, step: int, name: str, value: float
    ) -> None:
        """Writes metric to WandB."""
        self.run.log({name: value, "epoch": epoch}, step=step)

    def after_write_metric(self, epoch: int, step: int) -> None:
        """Callback executed after writing metric."""

    def save_model(self, epoch: int, algo: SaveProtocol) -> None:
        """Saves models to Weights & Biases.

        """
        # Implement saving model to wandb if needed
        save_name = f"{self.save_dir}/model_last.d3"
        algo.save_model(save_name)
        print(f"Model saved to {save_name}")

    def close(self) -> None:
        """Closes the logger and finishes the WandB run."""
        self.run.finish()


class WanDBAdapterFactory(LoggerAdapterFactory):
    r"""WandB Logger Adapter Factory class.

    This class creates instances of the WandB Logger Adapter for experiment
    tracking.
    """

    _project: Optional[str]

    def __init__(self, project: Optional[str] = None) -> None:
        """Initialize the WandB Logger Adapter Factory.

        Args:
            project (Optional[str], optional): The name of the WandB project. Defaults to None.
        """
        self._project = project

    def create(self, experiment_name: str) -> LoggerAdapter:
        """Creates a WandB Logger Adapter instance.

        Args:
            experiment_name (str): Name of the experiment.

        Returns:
            Instance of the WandB Logger Adapter.
        """
        return WanDBAdapter(
            experiment_name=experiment_name,
            project=self._project,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str,
                        default="./trajectories/PST_V2G_ProfixMax_25_optimal_25_100.pkl.gz",
                        help="Path to .npy file with trajectories")
    parser.add_argument("--algo", type=str,  default="bc",
                        help="Offline RL algorithm to use")
    parser.add_argument("--n_steps", type=int, default=100_000,
                        help="Total training steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str,
                        choices=["cpu", "cuda"], default="cuda")

    # WandB
    parser.add_argument("--wandb_project", type=str, default="DT4EVs",
                        help="WandB project name")
    parser.add_argument("--wandb_run", type=str, default="run-1",
                        help="WandB run name")

    parser.add_argument('--config_file', type=str,
                        default="PST_V2G_ProfixMax_25.yaml")

    parser.add_argument('--num_eval_episodes', type=int, default=30)
    parser.add_argument('--eval_replay_path', type=str,
                        default="./eval_replays/PST_V2G_ProfixMax_25_optimal_25_50/")

    args = parser.parse_args()
    main(args)
