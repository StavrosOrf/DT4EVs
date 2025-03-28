'''Find the best reward for each dataset and algorithm together with the corresponding epoch'''

    # results = {
    #     "algorithm": algorithm,
    #     "K": K,
    #     "dataset": dataset,
    #     "seed": seed,
    #     "best": np.array(history["best"])[-1],
    #     "best_reward": np.array(history["best"]),
    #     "eval_reward": np.array(history["test/total_reward"]),
    #     "eval_profits": np.array(history["test/total_profits"]),
    #     "eval_power_tracker_violation": np.array(history["test/power_tracker_violation"]),
    #     "eval_user_satisfaction": np.array(history["test/average_user_satisfaction"]),
    #     "opt_reward": np.array(history["opt/total_reward"])[-1],
    #     "opt_profits": np.array(history["opt/total_profits"])[-1],
    #     "opt_power_tracker_violation": np.array(history["opt/power_tracker_violation"])[-1],
    #     "opt_user_satisfaction": np.array(history["opt/average_user_satisfaction"])[-1],
    # }   
    
# # Plot the results

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from res_utils import dataset_info, parse_string_to_list


data = pd.read_csv("./results_analysis/results.csv")
dataset_info(data)

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

dataset_info(data)

# For every row in the data create a new dataframe with the best reward for each dataset and algorithm, and K together with the corresponding epoch

new_df = pd.DataFrame()

for i, row in data.iterrows():    
    rewards = parse_string_to_list(row["eval_reward"])
    profits = parse_string_to_list(row["eval_profits"])
    user_satisfaction = parse_string_to_list(row["eval_user_satisfaction"])
    power_tracker_violation = parse_string_to_list(row["eval_power_tracker_violation"])
    # print(f'rewards: {rewards[:10]}')
    # print(f'profits: {profits[:10]}')
    # print(f'user_satisfaction: {user_satisfaction[:10]}')
    # print(f'power_tracker_violation: {power_tracker_violation[:10]}')
    # input()
    # max reward index, max reward
    max_reward_index = np.argmax(rewards)
    max_reward = rewards[max_reward_index]
    
    new_df = pd.concat([new_df,
                        pd.DataFrame({
                            "algorithm": row["algorithm"],
                            "K": row["K"],
                            "dataset": row["dataset"],
                            "seed": row["seed"],
                            "max_reward": max_reward/100000,
                            "profits": profits[max_reward_index],
                            "user_satisfaction": user_satisfaction[max_reward_index],
                            "power_violation": power_tracker_violation[max_reward_index],
                            "runtime": row["runtime"],
                            "epochs": len(rewards),
                            "max_reward_epoch": max_reward_index
                        }, index=[0])], ignore_index=True)
print(new_df)
    # exit()
# group the data by algorithm, K, and dataset and show the max max_reward for each group
# grouped = new_df.groupby(["algorithm", "K", "dataset"]).max()
grouped_max = new_df.loc[new_df.groupby(["algorithm", "K", "dataset"])["max_reward"].idxmax()]
print(grouped_max)
# print(grouped)
#save the grouped data to a csv file
grouped_max.to_csv("./results_analysis/max_rewards.csv", index=False)
