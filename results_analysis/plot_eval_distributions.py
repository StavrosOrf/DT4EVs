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
import seaborn as sns
import warnings
from matplotlib.ticker import MaxNLocator

# Suppress all UserWarnings
# warnings.filterwarnings("ignore", category=UserWarning)

# set seeds
seed = 9  # 6
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)


def evaluator(config_files):

    ############# Simulation Parameters #################
    n_test_cycles = 100

    # save the list of EV profiles to a pickle file
    save_path = './results_analysis/scenarios/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for config_file in config_files:
        ev_profiles = []
        power_limit = []

        for i in range(n_test_cycles):
            env = EV2Gym(config_file=config_file,
                         generate_rnd_game=True,
                         save_replay=False,
                         )

            for _ in range(env.simulation_length):
                actions = np.ones(env.cs)

                new_state, reward, done, truncated, _ = env.step(
                    actions, visualize=False)  # takes action

                if done:
                    break

            env.power_setpoints = env.power_setpoints[:env.simulation_length]
            ev_profiles.append(env.EVs_profiles)

            power_limit.append(env.power_setpoints)

        # save the list of EV profiles to a pickle file
        scenario = config_file.split("/")[-1].split(".")[0]
        with open(os.path.join(save_path, f"{scenario}_ev_profiles.pkl"), "wb") as f:
            pickle.dump(ev_profiles, f)

        with open(os.path.join(save_path, f"{scenario}_power_limit.pkl"), "wb") as f:
            pickle.dump(power_limit, f)


def process_data(config_files):

    plt.rcParams.update({'font.size': 12})
    # plt.rcParams['font.family'] = ['serif']
    plt.rcParams['font.family'] = 'serif'

    plt.figure(figsize=(10, 5))

    for scenario in config_files:
        scenario = scenario.split("/")[-1].split(".")[0]
        print(f"Processing {scenario}...")

        # load the list of power limits from a pickle file
        with open(f"./results_analysis/scenarios/{scenario}_power_limit.pkl", "rb") as f:
            power_limit = pickle.load(f)

        # make power limit to a 2d array
        power_limit = np.array(power_limit)
        power_limit = power_limit.reshape(-1, power_limit.shape[-1])

        # plot average and std of the power limit for every step (3000 steps)

        plt.plot(np.mean(power_limit, axis=0), label=f'{scenario}')
        plt.fill_between(range(power_limit.shape[1]),
                         np.mean(power_limit, axis=0) -
                         np.std(power_limit, axis=0),
                         np.mean(power_limit, axis=0) +
                         np.std(power_limit, axis=0),
                         alpha=0.2,)

    # plt.title(f"Power Limit for {scenario}",fontsize=12)
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Power Limit (kW)", fontsize=12)
    plt.ylim(0, 250)
    plt.xlim(0, 300)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"./results_analysis/scenarios/power_limit.png")


def plot_scenario_distributions(config_files):

    # Simulation Starting Time
    # Hour and minute do not change after the environment has been reset
    hour = 5  # Simulation starting hour (24 hour format)
    minute = 0  # Simulation starting minute (0-59)

    timescale = 15  # in minutes per step
    simulation_length = 300

    all_data = pd.DataFrame()

    for index, ev_profile_file in enumerate(config_files):

        scenario = ev_profile_file.split("/")[-1].split(".")[0]
        # load the list of power limits from a pickle file
        with open(f"./results_analysis/scenarios/{scenario}_ev_profiles.pkl", "rb") as f:
            ev_profiles = pickle.load(f)

        if index == 0:
            case_name = "Original"
        elif index == 1:
            case_name = "Small"
        elif index == 2:
            case_name = "Medium"
        elif index == 3:
            case_name = "Extreme"

        EVs = {"arrival_time": [],
               "departure_time": [],
               "SoC_at_arrival": [],
               "battery_capacity": [],
               "charging_power": [],
               "time_of_stay": [],
               "case": case_name
               }

        for EV in ev_profiles[0]:
            # print(EV)

            arrival_time = EV.time_of_arrival * timescale + hour * 60 + minute
            departure_time = EV.time_of_departure * timescale + hour * 60 + minute

            arrival_time = int(arrival_time % 1440)
            if arrival_time > 1440:
                arrival_time = arrival_time - 1440

            departure_time = int(departure_time % 1440)
            if departure_time > 1440:
                departure_time = departure_time - 1440

            SoC_at_arrival = (EV.battery_capacity_at_arrival /
                              EV.battery_capacity) * 100
            battery_capacity = EV.battery_capacity
            charging_power = EV.max_ac_charge_power
            time_of_stay = (EV.time_of_departure -
                            EV.time_of_arrival)*timescale / 60

            EVs["arrival_time"].append(arrival_time)
            EVs["departure_time"].append(departure_time)
            EVs["SoC_at_arrival"].append(SoC_at_arrival)
            EVs["battery_capacity"].append(battery_capacity)
            EVs["charging_power"].append(charging_power)
            EVs["time_of_stay"].append(time_of_stay)

            # print(EVs)
            # exit()

        data = pd.DataFrame(EVs)
        all_data = pd.concat([all_data, data])

    # Create the figure for subplots
    # figure, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot the distribution of SoC at arrival

    # plt.subplot(1, 3, 1)
    plt.rcParams.update({'font.size': 12})
    # plt.rcParams['font.family'] = ['serif']
    plt.rcParams['font.family'] = 'serif'

    g0 = sns.JointGrid(data=all_data,
                       x="time_of_stay",
                       y="arrival_time",
                       hue="case",
                       xlim=(0, 20),
                       ylim=(200, 1600),
                       ratio=3,
                       height=4,
                       )

    # Plotting the joint KDE plot with transparency
    g0.plot_joint(sns.kdeplot,
                  levels=20,
                  #   gridsize=1000,
                  thresh=0.35,
                  #   cut=5,
                  fill=True,
                  alpha=0.7,
                  #   cummulative=True,
                  common_grid=True,
                  common_norm=True,
                  )

    # Plotting the marginal KDE plots with transparency
    g0.plot_marginals(sns.kdeplot, alpha=1, common_norm=False)
    # g0.ax_marg_y.tick_params(labeltop=True,
    #                          rotation=90,
    #                          labelsize=10,
    #                          )
    # g0.ax_marg_y.grid(True, axis='x', ls='--')
    g0.ax_marg_y.spines['top'].set_visible(True)
    g0.ax_marg_y.spines['right'].set_visible(True)
    g0.ax_marg_y.spines['bottom'].set_visible(True)
    # g0.ax_marg_y.xaxis.set_major_locator(MaxNLocator(4))

    # g0.ax_marg_x.tick_params(labelleft=True,
    #                          labelsize=10,
    #                          )
    # g0.ax_marg_x.grid(True, axis='y', ls='--')

    g0.ax_marg_x.spines['top'].set_visible(True)
    g0.ax_marg_x.spines['bottom'].set_visible(True)
    g0.ax_marg_x.spines['left'].set_visible(True)
    g0.ax_marg_x.spines['right'].set_visible(True)
    # g0.ax_marg_x.xaxis.set_major_locator(MaxNLocator(4))

    g0.ax_joint.legend().remove()

    # plt.subplot(1, 3, 2)

    g1 = sns.JointGrid(data=all_data,
                       x="SoC_at_arrival",
                       y="arrival_time",
                       hue="case",
                       xlim=(0, 105),
                       ylim=(200, 1600),
                       ratio=3,
                       #  kind="kde",
                       #  palette="viridis",
                       height=4,
                       #  ratio=5
                       )

    # Plotting the joint KDE plot with transparency
    g1.plot_joint(sns.kdeplot,
                  levels=20,
                  #   gridsize=1000,
                  thresh=0.2,
                  #   cut=5,
                  fill=True,
                  alpha=0.7,
                  #   cummulative=True,
                  common_grid=True,
                  common_norm=True,
                  )

    # Plotting the marginal KDE plots with transparency
    g1.plot_marginals(sns.kdeplot, alpha=1, common_norm=False)

    # g1.ax_marg_y.tick_params(labeltop=True,
    #                          rotation=90,
    #                          labelsize=10,
    #                          )

    # g1.ax_marg_y.grid(True, axis='x', ls='--')

    g1.ax_marg_y.spines['top'].set_visible(True)
    g1.ax_marg_y.spines['left'].set_visible(True)
    g1.ax_marg_y.spines['right'].set_visible(True)
    g1.ax_marg_y.spines['bottom'].set_visible(True)
    # g1.ax_marg_y.xaxis.set_major_locator(MaxNLocator(4))

    # g1.ax_marg_x.tick_params(labelleft=True,
    #                          labelsize=10,
    #                          )
    # g1.ax_marg_x.grid(True, axis='y', ls='--')
    g1.ax_marg_x.spines['top'].set_visible(True)
    g1.ax_marg_x.spines['bottom'].set_visible(True)
    g1.ax_marg_x.spines['left'].set_visible(True)
    g1.ax_marg_x.spines['right'].set_visible(True)
    # g1.ax_marg_x.xaxis.set_major_locator(MaxNLocator(4))

    # g1.ax_joint.legend()
    # remove the legend
    # g1.ax_joint.legend().remove()

    # add legend outside the plot

    g2 = sns.JointGrid(data=all_data,
                       y="departure_time",
                       x="SoC_at_arrival",
                       hue="case",
                       xlim=(0, 105),
                       ylim=(200, 1600),
                       height=4,
                       ratio=3,
                       #  kind="reg",
                       #  palette="Paired",
                       # height=6,
                       # ratio=5
                       )

    # Plotting the joint KDE plot with transparency
    g2.plot_joint(sns.kdeplot,
                  levels=20,
                  #   gridsize=1000,
                  thresh=0.3,
                  #   cut=5,
                  fill=True,
                  alpha=0.7,
                  #   cummulative=True,
                  common_grid=True,
                  common_norm=True,
                  )

    # Plotting the marginal KDE plots with transparency
    g2.plot_marginals(sns.kdeplot, alpha=1, common_norm=False
                      )

    # normalize the y-axis
    g2.ax_marg_y.set_ylim(0, 1600)

    # g2.ax_marg_y.tick_params(labeltop=True,
    #                         rotation=90,
    #                         labelsize=10,
    #                         )
    # g2.ax_marg_y.grid(True, axis='x', ls='--')
    g2.ax_marg_y.spines['top'].set_visible(True)
    g2.ax_marg_y.spines['left'].set_visible(True)
    g2.ax_marg_y.spines['right'].set_visible(True)
    g2.ax_marg_y.spines['bottom'].set_visible(True)

    # g2.ax_marg_y.xaxis.set_major_locator(MaxNLocator(4))

    # g2.ax_marg_x.tick_params(labelleft=True,
    #                          labelsize=10,
    #                          )

    # g2.ax_marg_x.grid(True, axis='y', ls='--')
    g2.ax_marg_x.spines['top'].set_visible(True)
    g2.ax_marg_x.spines['bottom'].set_visible(True)
    g2.ax_marg_x.spines['left'].set_visible(True)
    g2.ax_marg_x.spines['right'].set_visible(True)
    g2.ax_marg_x.xaxis.set_major_locator(MaxNLocator(4))

    g0.ax_marg_y.grid(True, axis='y', ls='--')
    g1.ax_marg_y.grid(True, axis='y', ls='--')
    g2.ax_marg_y.grid(True, axis='y', ls='--')

    # set the y-axis to scientific notation
    # g0.ax_marg_x.ticklabel_format(axis='y',
    #                               style='sci',
    #                               scilimits=(-2, -2))

    # g0.ax_marg_y.xaxis.get_offset_text().set_fontsize(9)
    # g0.ax_marg_x.yaxis.get_offset_text().set_fontsize(9)

    # g0.ax_marg_y.ticklabel_format(axis='x',
    #                               style='sci',
    #                               scilimits=(-4, -4))

    # set the y-axis to scientific notation
    # g1.ax_marg_x.ticklabel_format(axis='y',
    #                               style='sci',
    #                               scilimits=(-3, -3))
    # g1.ax_marg_y.ticklabel_format(axis='x',
    #                               style='sci',
    #                               scilimits=(-4, -4))
    # g1.ax_marg_y.xaxis.get_offset_text().set_fontsize(9)
    # g1.ax_marg_x.yaxis.get_offset_text().set_fontsize(9)

    # set the y-axis to scientific notation
    # g2.ax_marg_x.ticklabel_format(axis='y',
    #                               style='sci',
    #                               scilimits=(-3, -3))
    # g2.ax_marg_y.ticklabel_format(axis='x',
    #                               style='sci',
    #                               scilimits=(-5, -5))

    # g2.ax_marg_y.xaxis.get_offset_text().set_fontsize(9)
    # g2.ax_marg_x.yaxis.get_offset_text().set_fontsize(9)

    # g0.ax_marg_x.set_title('Marginal PDFs [-]', fontsize=11)
    # g1.ax_marg_x.set_title('Marginal PDFs [-]', fontsize=11)
    # g2.ax_marg_x.set_title('Marginal PDFs [-]', fontsize=11)

    g0.ax_marg_x.grid(True, axis='x', ls='--')
    g1.ax_marg_x.grid(True, axis='x', ls='--')
    g2.ax_marg_x.grid(True, axis='x', ls='--')

    g2.ax_joint.legend().remove()
    g1.ax_joint.legend().remove()

    # sns.move_legend(g1.ax_joint,
    #                 "lower left",
    #                 title=None,
    #                 #number of columns
    #                 ncol=1,
    #                 frameon=True,
    #                 #width of the legend line
    #                 handlelength=1,
    #                 fontsize=11,)

    # remove the y-axis labels
    g0.ax_joint.set_ylabel('Arrival Time [Hour]', fontsize=11)

    g0.ax_joint.set_yticks([0, 360, 720, 1080, 1440],
                           ['00:00', '06:00', '12:00', '18:00', '24:00'],
                           fontsize=11)

    g1.ax_joint.set_yticks([0, 360, 720, 1080, 1440],
                           ['00:00', '06:00', '12:00', '18:00', '24:00'],
                           fontsize=11)

    g2.ax_joint.set_yticks([0, 360, 720, 1080, 1440],
                           ['00:00', '06:00', '12:00', '18:00', '24:00'],
                           fontsize=11)

    # add grid lines
    g0.ax_joint.grid(axis='y', linestyle='--', alpha=0.5)
    g1.ax_joint.grid(axis='y', linestyle='--', alpha=0.5)
    g2.ax_joint.grid(axis='y', linestyle='--', alpha=0.5)

    g0.ax_joint.grid(axis='x', linestyle='--', alpha=0.5)
    g1.ax_joint.grid(axis='x', linestyle='--', alpha=0.5)
    g2.ax_joint.grid(axis='x', linestyle='--', alpha=0.5)

#   change font size of x-ticks
    g0.ax_joint.set_xticks([0, 5, 10, 15, 20],
                           ['0', '5', '10', '15', '20'],
                           fontsize=11)
    g1.ax_joint.set_xticks([0, 20, 40, 60, 80, 100],
                           ['0', '20', '40', '60', '80', '100'],
                           fontsize=11)

    g2.ax_joint.set_xticks([0, 20, 40, 60, 80, 100],
                           ['0', '20', '40', '60', '80', '100'],
                           fontsize=11)

    g0.ax_joint.set_xlabel('Time of Stay [Hour]', fontsize=11)
    # g1.ax_joint.set_ylabel('')
    g1.ax_joint.set_ylabel('Arrival Time [Hour]', fontsize=11)
    g1.ax_joint.set_xlabel('SoC at Arrival [%]', fontsize=11)
    g2.ax_joint.set_ylabel('Departure Time [Hour]', fontsize=11)
    g2.ax_joint.set_xlabel('SoC at Arrival [%]', fontsize=11)

    # save each subplot as a separate figure pdf
    # save g0 as a separate figure
    # fig = plt.figure(figsize=(3.3, 4))
    # gs = gridspec.GridSpec(1, 1)

    # mg0 = SeabornFig2Grid(g0, fig, gs[0])
    # plt.tight_layout()
    # plt.savefig("./Results_Analysis/figures/scenario_distributions_g0.pdf")

    # plot g0 using seaborn as pdf
    save_path = './results_analysis/scenarios/'
    g0.savefig(f"{save_path}/scenario_distributions_g0.pdf")
    g1.savefig(f"{save_path}/scenario_distributions_g1.pdf")
    g2.savefig(f"{save_path}/scenario_distributions_g2.pdf")


def plot_scenario_distributions_2(config_files):

    # Simulation Starting Time
    # Hour and minute do not change after the environment has been reset
    hour = 5  # Simulation starting hour (24 hour format)
    minute = 0  # Simulation starting minute (0-59)

    timescale = 15  # in minutes per step
    simulation_length = 300

    all_data = pd.DataFrame()

    for index, ev_profile_file in enumerate(config_files):

        scenario = ev_profile_file.split("/")[-1].split(".")[0]
        # load the list of power limits from a pickle file
        with open(f"./results_analysis/scenarios/{scenario}_ev_profiles.pkl", "rb") as f:
            ev_profiles = pickle.load(f)

        if index == 0:
            case_name = "Original"
        elif index == 1:
            case_name = "Small"
        elif index == 2:
            case_name = "Medium"
        elif index == 3:
            case_name = "Extreme"

        EVs = {"arrival_time": [],
               "departure_time": [],
               "SoC_at_arrival": [],
               "battery_capacity": [],
               "charging_power": [],
               "time_of_stay": [],
               "case": case_name
               }

        for EV in ev_profiles[0]:
            # print(EV)

            arrival_time = EV.time_of_arrival * timescale + hour * 60 + minute
            departure_time = EV.time_of_departure * timescale + hour * 60 + minute

            arrival_time = int(arrival_time % 1440)
            if arrival_time > 1440:
                arrival_time = arrival_time - 1440

            departure_time = int(departure_time % 1440)
            if departure_time > 1440:
                departure_time = departure_time - 1440

            SoC_at_arrival = (EV.battery_capacity_at_arrival /
                              EV.battery_capacity) * 100
            battery_capacity = EV.battery_capacity
            charging_power = EV.max_ac_charge_power
            time_of_stay = (EV.time_of_departure -
                            EV.time_of_arrival)*timescale / 60

            EVs["arrival_time"].append(arrival_time)
            EVs["departure_time"].append(departure_time)
            EVs["SoC_at_arrival"].append(SoC_at_arrival)
            EVs["battery_capacity"].append(battery_capacity)
            EVs["charging_power"].append(charging_power)
            EVs["time_of_stay"].append(time_of_stay)

            # print(EVs)
            # exit()

        data = pd.DataFrame(EVs)
        all_data = pd.concat([all_data, data])

    # plot the histogram of the SoC at arrival
    plt.figure(figsize=(5, 5))
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['font.family'] = ['serif']

    sns.kdeplot(
        data=all_data,
        x="SoC_at_arrival",
        hue="case",
        common_norm=False,
        fill=True,
        alpha=0.5,
        linewidth=2,
    )

    plt.xlabel("SoC at Arrival [%]", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    # plt.title("SoC at Arrival", fontsize=12)
    plt.gca().legend_.set_title("Scenario")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"./results_analysis/scenarios/SoC_at_arrival.png")
    plt.savefig(f"./results_analysis/scenarios/SoC_at_arrival.pdf")

    # plot the histogram of the arrival time
    plt.figure(figsize=(5, 5))

    sns.kdeplot(
        data=all_data,
        x="arrival_time",
        hue="case",
        common_norm=False,
        fill=True,
        alpha=0.5,
        linewidth=2,
    )
    plt.xlabel("Arrival Time [Hour]", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.grid()
    # plt.gca().legend_.set_title("Scenario")
    # plt.gca().legend(title="Scenario", loc='upper left')
    plt.tight_layout()    
    plt.savefig(f"./results_analysis/scenarios/arrival_time.png")
    plt.savefig(f"./results_analysis/scenarios/arrival_time.pdf")

    # plot the histogram of the departure time
    plt.figure(figsize=(5, 5))
    sns.kdeplot(
        data=all_data,
        x="departure_time",
        hue="case",
        common_norm=False,
        fill=True,
        alpha=0.5,
        linewidth=2,
    )
    plt.xlabel("Departure Time [Hour]", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.grid()
    plt.tight_layout()
    plt.gca().legend_.set_title("Scenario")
    plt.savefig(f"./results_analysis/scenarios/departure_time.png")
    plt.savefig(f"./results_analysis/scenarios/departure_time.pdf")

    # plot the histogram of the time of stay
    plt.figure(figsize=(5, 5))
    sns.kdeplot(
        data=all_data,
        x="time_of_stay",
        hue="case",
        common_norm=False,
        fill=True,
        alpha=0.5,
        linewidth=2,
    )
    plt.xlabel("Time of Stay [Hour]", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.gca().legend_.set_title("Scenario")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"./results_analysis/scenarios/time_of_stay.png")
    plt.savefig(f"./results_analysis/scenarios/time_of_stay.pdf")


if __name__ == "__main__":

    # load the list of EV profiles from a pickle file
    config_files = [
        "./config_files/PST_V2G_ProfixMax_25.yaml",
        "./config_files/PST_V2G_ProfixMax_25_G1.yaml",
        "./config_files/PST_V2G_ProfixMax_25_G2.yaml",
        "./config_files/PST_V2G_ProfixMax_25_G3.yaml",
    ]

    # config_files = [
    #     "./config_files/PST_V2G_ProfixMax_25_CS5.yaml",
    #     "./config_files/PST_V2G_ProfixMax_25.yaml",
    #     "./config_files/PST_V2G_ProfixMax_25_CS50.yaml",
    #     "./config_files/PST_V2G_ProfixMax_25_CS75.yaml",
    #     "./config_files/PST_V2G_ProfixMax_25_CS100.yaml",
    # ]

    # evaluator(config_files)
    # process_data(config_files)
    plot_scenario_distributions_2(config_files)
    # plot_scenario_distributions(config_files)
