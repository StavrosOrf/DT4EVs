import wandb
import pandas as pd
import numpy as np
import tqdm as tqdm
# Login to W&B if not already logged in
# wandb.login()

# Initialize API
api = wandb.Api(timeout=120)

# Replace 'your_project_name' and 'your_entity_name' with your actual project and entity
project_name = "DT4EVs"
# group_name = "PaperExpsDT_25cs"
group_name = "25_SB3_RL_tests"
# group_name = "PaperExpsDT_250cs"

# Fetch runs from the specified project
runs = api.runs(f"{entity_name}/{project_name}")
print(f"Total runs fetched: {len(runs)}")

runs = [run for run in runs if run.group == group_name]
print(f"Total GNN runs fetched: {len(runs)}")

# Display the filtered runs with group names

run_results = []
# use tqdm to display a progress bar
for i, run in tqdm.tqdm(enumerate(runs), total=len(runs)):
    # if i != 23:
    #     continue
    # print(run.name)
    group_name = run.group
    history = run.history()
    mean_rewards = history["eval/mean_reward"]
    mean_rewards = [None if np.isnan(x) else x for x in mean_rewards]
    mean_rewards = [x for x in mean_rewards if x != None]
    print(f'{run.name} mean_rewards: {len(mean_rewards)}')
    # continue
    # exit()
    
    name = run.name

    algorithm = name.split("_")[0]
        
    dataset = "online"
    seed = name.split("_")[-1]
    # print(history.keys())
    # print(f'history: {history}')
    results = {
        "algorithm": algorithm,
        "K": -1,
        "dataset": dataset,
        "seed": seed,
        "runtime": np.array(history["_runtime"])[-1]/3600,
        "best": max(mean_rewards),
        # "best_reward": np.array(history["eval/best_reward"]),
        "eval_reward": np.array(mean_rewards),
        # "eval_profits": np.array(history["test/total_profits"]),
        # "eval_power_tracker_violation": np.array(history["test/power_tracker_violation"]),
        # "eval_user_satisfaction": np.array(history["test/average_user_satisfaction"]),
        # "opt_reward": np.array(history["opt/total_reward"])[-1],
        # "opt_profits": np.array(history["opt/total_profits"])[-1],
        # "opt_power_tracker_violation": np.array(history["opt/power_tracker_violation"])[-1],
        # "opt_user_satisfaction": np.array(history["opt/average_user_satisfaction"])[-1],
    }   
    # print(f'results_rewards: {results["eval_reward"][:40]}')
    # print(f'results_profits: {results["eval_profits"][:40]}')
    # print(f'results_power_tracker_violation: {results["eval_power_tracker_violation"][:40]}')
    # print(f'results_user_satisfaction: {results["eval_user_satisfaction"][:40]}')
    # input("Press")
    
    run_results.append(results)
    # exit()
    
    # if i > 102:
    #     break
    
# Convert the results to a pandas DataFrame
df = pd.DataFrame(run_results)
print(df.head())
print(df.shape)

print(df.describe())

print(df["algorithm"].value_counts())
print(df["dataset"].value_counts())
print(df["seed"].value_counts())

# Save the results to a CSV file
df.to_csv("./results_analysis/resultsClassicRL25.csv", index=False)
print("Results saved to results.csv")
