import wandb
import pandas as pd
import numpy as np
import tqdm as tqdm
# Login to W&B if not already logged in
# wandb.login()

# Initialize API
api = wandb.Api()

# Replace 'your_project_name' and 'your_entity_name' with your actual project and entity
project_name = "DT4EVs"
group_name = "PaperExpsDT_25cs"
# group_name = "25_ClassicRL_tests"
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
    # if i < 100:
        # continue
    group_name = run.group
    history = run.history()
    
    # print(f"History keys: {history.keys()}")
    # print(f"_runtime: {history['_runtime']}")
    # print(f"_timestamp: {history['_timestamp']}")
    # print(f"_step: {history['_step']}")
    # exit()
    # History keys: Index(['best', 'time/total', 'time/training', 'opt/total_transformer_overload',
    #    'test/total_energy_charged', '_step', 'time/evaluation',
    #    'test/total_profits', 'opt/average_user_satisfaction', '_timestamp',     
    #    'training/action_error', 'opt/total_reward',
    #    'opt/power_tracker_violation', 'test/total_transformer_overload',        
    #    'test/power_tracker_violation', '_runtime',
    #    'test/average_user_satisfaction', 'training/train_loss_std',
    #    'training/train_loss_mean', 'test/min_user_satisfaction',
    #    'opt/min_user_satisfaction', 'opt/total_profits', 'test/total_reward',   
    #    'opt/total_energy_charged', 'opt/total_energy_discharged',
    #    'test/total_energy_discharged'],
    #   dtype='object')
#  clarify the algorithm used in the run, and the dataset used
    config = run.config
    # print(f'config: {config}')
    if "model_type" not in config:
        if "QT" not in config["name"]:
            raise ValueError("Model type not found in config")
        else:
            algorithm = "QT"
    else:
        algorithm = config["model_type"]
        
    K = config["K"]
    dataset = config["dataset"]
    seed = config["seed"]
    
    if '_runtime' not in history:
        print(f"Run {run.id} has no _runtime key")
        continue
    
    results = {
        "algorithm": algorithm,
        "K": K,
        "dataset": dataset,
        "seed": seed,
        "runtime": np.array(history["_runtime"])[-1]/3600,
        "best": np.array(history["best"])[-1],
        "best_reward": np.array(history["best"]),
        "eval_reward": np.array(history["test/total_reward"]),
        "eval_profits": np.array(history["test/total_profits"]),
        "eval_power_tracker_violation": np.array(history["test/power_tracker_violation"]),
        "eval_user_satisfaction": np.array(history["test/average_user_satisfaction"]),
        "opt_reward": np.array(history["opt/total_reward"])[-1],
        "opt_profits": np.array(history["opt/total_profits"])[-1],
        "opt_power_tracker_violation": np.array(history["opt/power_tracker_violation"])[-1],
        "opt_user_satisfaction": np.array(history["opt/average_user_satisfaction"])[-1],
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
print(df["K"].value_counts())
print(df["dataset"].value_counts())
print(df["seed"].value_counts())

# Save the results to a CSV file
df.to_csv("./results_analysis/results.csv", index=False)
print("Results saved to results.csv")
