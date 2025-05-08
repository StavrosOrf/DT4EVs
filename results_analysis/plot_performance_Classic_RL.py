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

data = pd.read_csv("./results_analysis/resultsClassicRL25.csv")
# dataset_info(data)
print(data.columns.tolist())


# # filter the data that have:
# dataset_info(data)

# For every row in the data create a new dataframe with epoch as the index and the reward as the value, keep also, the seed, algorithm and dataset

new_df = pd.DataFrame()
for i, row in data.iterrows():
    # parse the string to a list
    rewards = parse_string_to_list(row["eval_reward"])
    
    for j in range(len(rewards)):
        reward = rewards[j] if j < len(rewards) else rewards[-1]
        entry = {
            "epoch": j,
            "reward": reward,     
            "seed": row["seed"],
            "algorithm": row["algorithm"],
            "dataset": row["dataset"]
        }
        new_df = pd.concat([new_df, pd.DataFrame([entry])])
# print(new_df.head())    
print(new_df.describe())
    
datasets_list = [
    'online',
]
# if algorithm is  a2c remove it
new_df = new_df[new_df["algorithm"] != "a2c"]

# change algorithm names
# print unique values of the algorithm column
print(new_df["algorithm"].unique())
# from dt to DT
new_df["algorithm"] = new_df["algorithm"].replace("td3", "TD3")
# from QT to Q-DT
new_df["algorithm"] = new_df["algorithm"].replace("sac", "SAC")
# from gnn_act_emb to GNN-DT
new_df["algorithm"] = new_df["algorithm"].replace("tqc", "TQC")
new_df["algorithm"] = new_df["algorithm"].replace("ppo", "PPO")
new_df["algorithm"] = new_df["algorithm"].replace("trpo", "TRPO")
new_df["algorithm"] = new_df["algorithm"].replace("ddpg", "DDPG")

# plot the data
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.figure(figsize=(4, 4))
# ax = sns.lineplot(data=new_df,
#                 x="epoch",
#                 y="reward",
#                 # colors=["blue", "green", "red", "orange"],
#                 hue="algorithm",
#               

markers = {
    "TD3":  "o",
    "SAC":  "s",
    "TQC":  "^",
    "PPO":  "v",
    "TRPO": "P",
    "DDPG": "X",
}

cb = sns.color_palette("colorblind", 6)
set2 = sns.color_palette("Set2",       6)

# choose which one you like:
palette = cb      # or palette = set2

ax = sns.lineplot(
    data=new_df,
    x="epoch", y="reward",
    hue="algorithm",
    style="algorithm",
    palette=palette,
    markers=markers,
    dashes=False,      # solid lines
    legend="full"      # show color+marker
)

handles, labels = ax.get_legend_handles_labels()
#add a horizontal line for the optimal reward
plt.axhline(y=-2405, color='r', linestyle='--',
            label="Oracle")

# create a new legend for the optimal reward and the algorithms
# plt.legend(loc='center right',
#             title="Algorithm",
#             title_fontsize=14,
#             ncol=2,
#             columnspacing=0.4,
#             fontsize=13)

ax.get_legend().remove()
# plt.legend(loc='upper left')

# set x and y labels font size
plt.xlabel("Epoch", fontsize=17)
plt.ylabel("", fontsize=17)

#set xticks and yticks font size
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# put scientific notation in the y axis
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# set xlim
plt.xlim(0, 30)
# plt.ylim(-450_000, 10_000)
plt.tight_layout()
# plt.show()

plt.savefig(f"results_analysis/figs/plot_performance_classicRL.pdf",
            dpi=60)

plt.savefig(f"results_analysis/figs/plot_performance_onlineRL.png",
            dpi=60)

plt.clf()


fig_leg = plt.figure()
fig_leg.patch.set_facecolor('none')      # transparent background
leg = fig_leg.legend(
    handles, labels,
    loc='center',
    ncol=3,
    title=" ",
    # title_fontsize=15,
    fontsize=15.5,
    frameon=False
)
# 4. Save to file
fig_leg.savefig("results_analysis/figs/legend_1.png",
                bbox_inches='tight', dpi=60, transparent=True)
fig_leg.savefig("results_analysis/figs/legend_1.pdf",
                bbox_inches='tight', dpi=100, transparent=True)

plt.close(fig_leg)

