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


def preprocess_wandb_offlineRL():

    data = pd.read_csv("./results_analysis/offlineRL.csv")

    # Remove the columns that have a column name that contains "MIN" or "MAX"
    data = data.loc[:, ~data.columns.str.contains("MIN|MAX")]

    # print(data.columns)
    # limit to 250 rows
    data = data.iloc[:150, :]
    print(f'Offline RL data shape: {data.shape}')

    # make into dataframe with the columns:
    # ['epoch', 'reward', 'seed', 'algorithm', 'dataset']
    new_df = pd.DataFrame()

    for i, row in data.iterrows():

        # print(row)
        # get column names
        columns = data.columns

        # for every value in the row, if the value is not a string, add it to the new_df
        for j in range(1, len(row)):
            if j < 1:
                continue
            # print(f'row[j]: {row[j]}')
            name = columns[j]

            dataset = name.split("_")[1]
            algorithm = name.split("_")[0]
            seed = name.split("_")[3]
            # print(f' j: {j}| algorithm: {algorithm} | seed: {seed} | dataset: {dataset}')

            #     continue
            # print(row[j])
            entry = {
                "epoch": i,
                "reward": row[j],
                "seed": seed,
                "algorithm": algorithm,
                "dataset": f'{dataset}_10000'
            }
            # print(entry)
            new_df = pd.concat([new_df, pd.DataFrame([entry])])

    return new_df


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

# filter the data that have:
# data = data[(data["K"] == 2) & (data["dataset"].str.contains("optimal"))]
# data = data[(data["K"] == 10) & (data["dataset"].str.contains("optimal"))]
data = data[(data["K"] == 10)]
dataset_info(data)

# For every row in the data create a new dataframe with epoch as the index and the reward as the value, keep also, the seed, algorithm and dataset

new_df = pd.DataFrame()
for i, row in data.iterrows():
    # print(row)
    # parse the string to a list
    rewards = parse_string_to_list(row["eval_reward"])

    for j in range(250):
        # if there is no value for the epoch, use the last value

        reward = rewards[j] if j < len(rewards) else rewards[-1]
        entry = {
            "epoch": j,
            "reward": reward,
            "seed": row["seed"],
            "algorithm": row["algorithm"],
            "dataset": row["dataset"]
        }
        new_df = pd.concat([new_df, pd.DataFrame([entry])])


offlineRL_data = preprocess_wandb_offlineRL()

new_df = pd.concat([new_df, offlineRL_data])
# print(new_df.head())
# print(new_df.describe())

datasets_list = [
    'optimal_10000',
    'bau_10000',
    'random_10000',
]

# change algorithm names
# from dt to DT
new_df["algorithm"] = new_df["algorithm"].replace("dt", "DT")
new_df["algorithm"] = new_df["algorithm"].replace("iql", "IQL")
new_df["algorithm"] = new_df["algorithm"].replace("cql", "CQL")
new_df["algorithm"] = new_df["algorithm"].replace("bc", "BC")

# from QT to Q-DT
new_df["algorithm"] = new_df["algorithm"].replace("QT", "Q-DT")
# from gnn_act_emb to GNN-DT
new_df["algorithm"] = new_df["algorithm"].replace("gnn_act_emb", "GNN-DT")

# plot the data
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.figure(figsize=(4, 4))
for i in range(3):
    print(f"Plotting {datasets_list[i]}")
    # if datasets_list[i] == "optimal_10000":
    #     plt.figure(figsize=(4.4, 5.5))
    # else:
    #     continue
        # plt.figure(figsize=(4, 5))
    # plt.figure(figsize=(4, 5))
    ax = sns.lineplot(data=new_df[new_df["dataset"] == datasets_list[i]],
                 x="epoch",
                 y="reward",
                 hue="algorithm",
                 hue_order=["DT", "Q-DT", "GNN-DT",
                            "IQL", "CQL", "BC",],)

    # plt.title(f"K=10",
    #           fontsize=17)

    # add a horizontal line for the optimal reward
    plt.axhline(y=-2405, color='r', linestyle='--',
                label="Oracle")

    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    
    # rempove legend 
    # plt.legend_.remove()
    # create a new legend for the optimal reward and the algorithms
    # plt.legend(loc='lower right',
    #            title="Algorithm",
    #            title_fontsize=15,
    #            ncol=2,
    #            columnspacing=0.4,
    #            fontsize=14.5)
    
    # plt.legend(loc='upper left')

    # set x and y labels font size
    plt.xlabel("Epoch", fontsize=17)
    # if datasets_list[i] == "optimal_10000":
    #     plt.ylabel("Reward [-]", fontsize=17)
    # else:
    plt.ylabel("", fontsize=17)

    # set xticks and yticks font size
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    # put scientific notation in the y axis
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # set xlim
    plt.xlim(0, 150)
    plt.ylim(-450_000, 10_000)
    plt.tight_layout()

    plt.savefig(f"results_analysis/figs/plot_performance_{datasets_list[i]}.pdf",
                dpi=60)
    plt.savefig(f"results_analysis/figs/plot_performance_{datasets_list[i]}.png",
                dpi=60)
    
    #take the legend and plot it in a separate figure
    # plt.figure(figsize=(4, 5))
    plt.clf()

    if i == 0:
        fig_leg = plt.figure()
        fig_leg.patch.set_facecolor('none')      # transparent background
        leg = fig_leg.legend(
            handles, labels,
            loc='center',
            ncol=7,
            title=" ",
            # title_fontsize=15,
            fontsize=15.5,
            frameon=False
        )
        # 4. Save to file
        fig_leg.savefig("results_analysis/figs/legend.png",
                        bbox_inches='tight', dpi=60, transparent=True)
        fig_leg.savefig("results_analysis/figs/legend.pdf",
                        bbox_inches='tight', dpi=100, transparent=True)
        
        plt.close(fig_leg)
        plt.clf()
    # exit()
