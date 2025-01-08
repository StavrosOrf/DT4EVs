import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import math


def plot_comparable_EV_SoC(results_path,
                           save_path=None,
                           algorithm_names=None,
                           color_list=None,
                           marker_list=None
                           ):
    '''
    This function is used to plot the SoC of the EVs in the same plot
    '''

    with open(results_path, 'rb') as f:
        replay = pickle.load(f)

    plt.close('all')
    plt.figure(figsize=(15, 10))
    plt.rc('font', family='serif')

    for index, key in enumerate(replay.keys()):
        env = replay[key]

        color_list_map = plt.cm.get_cmap('Set1', len(replay.keys()))
        color_list = color_list_map(np.linspace(0, 1, len(replay.keys())))

        date_range = pd.date_range(start=env.sim_starting_date,
                                   end=env.sim_starting_date +
                                   (env.simulation_length - 1) *
                                   datetime.timedelta(
                                       minutes=env.timescale),
                                   freq=f'{env.timescale}min')

        date_range_print = pd.date_range(start=env.sim_starting_date,
                                         end=env.sim_date,
                                         periods=10)

        counter = 1
        dim_x = int(np.ceil(np.sqrt(env.cs)))
        dim_y = int(np.ceil(env.cs/dim_x))
        for cs in env.charging_stations:

            plt.subplot(dim_x, dim_y, counter)
            df = pd.DataFrame([], index=date_range)

            for port in range(cs.n_ports):
                df[port] = env.port_energy_level[port, cs.id, :]

            # Add another row with one datetime step to make the plot look better
            df.loc[df.index[-1] +
                   datetime.timedelta(minutes=env.timescale)] = df.iloc[-1]

            for port in range(cs.n_ports):
                for i, (t_arr, t_dep) in enumerate(env.port_arrival[f'{cs.id}.{port}']):
                    t_dep = t_dep + 1
                    if t_dep > len(df):
                        t_dep = len(df)
                    # x = df.index[t_arr:t_dep]
                    y = df[port].values.T[t_arr:t_dep]
                    # fill y with 0 before and after to match the length of df
                    y = np.concatenate(
                        [np.zeros(t_arr), y, np.zeros(len(df) - t_dep)])

                    if i == 0:
                        plt.step(df.index,
                                 y,
                                 where='post',
                                 color=color_list[index],
                                 marker=marker_list[index],
                                 label=algorithm_names[index]
                                 )
                    else:
                        plt.step(df.index,
                                 y,
                                 where='post',
                                 color=color_list[index],
                                 marker=marker_list[index]
                                 )

                    # change the marker spacing and distance
                    plt.setp(plt.gca().get_lines(), markersize=2)
                    plt.setp(plt.gca().get_lines(), markeredgewidth=0.5)

            plt.title(f'CS {cs.id + 1}', fontsize=12)

            if counter % dim_x == 1:
                plt.ylabel('SoC', fontsize=14)
                plt.yticks(np.arange(0.2, 1.1, 0.2),
                           fontsize=14)
            else:
                plt.ylabel('', fontsize=14)
                plt.yticks(np.arange(0.2, 1.1, 0.2),
                           labels=""*5,
                           fontsize=14)

            plt.ylim([0.2, 1.1])

            plt.xlim([env.sim_starting_date, env.sim_date])
            plt.grid(True, which='major', axis='both')

            if counter < 21:
                plt.xticks(ticks=date_range_print,
                           labels=""*9,
                           rotation=45,
                           fontsize=8)
            else:
                plt.xticks(ticks=date_range_print,
                           labels=[
                               f'{d.hour:2d}:{d.minute:02d}' for d in date_range_print],
                           rotation=45,
                           fontsize=8)

            counter += 1

    plt.legend(loc='upper center', bbox_to_anchor=(-1.75, -0.35),
               fancybox=True, shadow=True, ncol=5, fontsize=14)

    plt.grid(True, which='minor', axis='both')
    plt.tight_layout()
    plt.subplots_adjust(
        wspace=0.117,    # Width space between subplots
        hspace=0.279     # Height space between subplots
    )

    fig_name = f'{save_path}EV_Energy_Level.png'
    plt.savefig(fig_name, format='png',
                dpi=300, bbox_inches='tight')
    # plt.show()


def plot_comparable_EV_SoC_single(results_path,
                                  save_path=None,
                                  algorithm_names=None,
                                  color_list=None,
                                  marker_list=None
                                  ):
    '''
    This function is used to plot the SoC of the EVs in the same plot
    '''

    with open(results_path, 'rb') as f:
        replay = pickle.load(f)

    # plt.close('all')
    # fig, ax = plt.subplots()
    plt.figure(figsize=(6, 3.5))
    # get ax
    ax = plt.gca()

    plt.rc('font', family='serif')
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)

    # ax.spines['left'].set_linewidth(2)
    # ax.spines['bottom'].set_linewidth(2)

    plt.grid(True, which='major', axis='both',
             linestyle='--', linewidth=0.5)

    # plot a horizontal line at 0.8
    plt.axhline(y=0.8,
                color='blue',
                linestyle='--',
                alpha=0.5,
                label='Desired SoC')

    for index, key in enumerate(replay.keys()):
        env = replay[key]

        date_range = pd.date_range(start=env.sim_starting_date,
                                   end=env.sim_starting_date +
                                   (env.simulation_length - 1) *
                                   datetime.timedelta(
                                       minutes=env.timescale),
                                   freq=f'{env.timescale}min')

        date_range_print = pd.date_range(start=env.sim_starting_date,
                                         end=env.sim_date,
                                         periods=7)

        color_list_map = plt.cm.get_cmap('Set1', len(replay.keys()))
        color_list = color_list_map(np.linspace(0, 1, len(replay.keys())))

        cs_to_plot = 24
        counter = 1
        for cs in env.charging_stations:
            if counter != cs_to_plot:
                counter += 1
                continue

            # plt.subplot(1, 2, counter)
            df = pd.DataFrame([], index=date_range)

            for port in range(cs.n_ports):
                df[port] = env.port_energy_level[port, cs.id, :]

            # Add another row with one datetime step to make the plot look better
            df.loc[df.index[-1] +
                   datetime.timedelta(minutes=env.timescale)] = df.iloc[-1]

            for port in range(cs.n_ports):
                for i, (t_arr, t_dep) in enumerate(env.port_arrival[f'{cs.id}.{port}']):
                    t_dep = t_dep + 1
                    if t_dep > len(df):
                        t_dep = len(df)
                    # x = df.index[t_arr:t_dep]
                    y = df[port].values.T[t_arr:t_dep]
                    # fill y with 0 before and after to match the length of df
                    y = np.concatenate(
                        [np.zeros(t_arr), y, np.zeros(len(df) - t_dep)])

                    markevery = 5
                    if i == 0:
                        plt.step(df.index,
                                 y,
                                 where='post',
                                 color=color_list[index],
                                 marker=marker_list[index],
                                 markevery=markevery,
                                 alpha=0.8,
                                 label=algorithm_names[index])
                    else:
                        plt.step(df.index,
                                 y,
                                 where='post',
                                 color=color_list[index],
                                 marker=marker_list[index],
                                 markevery=markevery,
                                 alpha=0.8)

                    # change the marker spacing and distance
                    plt.setp(plt.gca().get_lines(), markersize=5)
                    plt.setp(plt.gca().get_lines(), markeredgewidth=0.5)

            # plt.title(f'Charging Station {cs.id + 1}', fontsize=24)

            plt.ylabel('SoC [-]', fontsize=14)
            plt.yticks(np.arange(0, 1.1, 0.2),
                       fontsize=14)

            # plt.xlabel('Time', fontsize=14)

            plt.ylim([0.1, 1.02])
            # plt.xlim([env.sim_starting_date, env.sim_date])
            # plt.xticks(ticks=date_range_print,
            #            labels=[f'{d.hour:2d}:{d.minute:02d}' for d in date_range_print],
            #            rotation=45,
            #            fontsize=14)

            starting_date = env.sim_starting_date + \
                datetime.timedelta(hours=16)
            ending_date = env.sim_starting_date + datetime.timedelta(hours=55)

            plt.xlim([starting_date, ending_date])
            date_range_print = pd.date_range(start=starting_date,
                                             end=ending_date,
                                             periods=7)

            plt.xticks(ticks=date_range_print,
                       labels=[
                           f'{d.hour:02d}:{d.minute:02d}' for d in date_range_print],
                       #    rotation=45,
                       fontsize=14
                       )
            counter += 1

            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
                       fancybox=True, shadow=True, ncol=3, fontsize=12)

    # plt.tight_layout()
    plt.subplots_adjust(
        left=0.143,    # Space from the left of the figure
        bottom=0.079,   # Space from the bottom of the figure
        right=0.933,   # Space from the right of the figure
        top=0.798,     # Space from the top of the figure
    )

    fig_name = f'{save_path}EV_Energy_Level_single.png'
    plt.savefig(fig_name, format='png',
                dpi=300,
                # bbox_inches='tight'
                )
    plt.savefig(f'{save_path}EV_Energy_Level_single.pdf',
                format='pdf',
                dpi=300,
                # bbox_inches='tight'
                )

    plt.show()


def plot_total_power_V2G(results_path, save_path=None, algorithm_names=None):

    # Load the env pickle files
    with open(results_path, 'rb') as f:
        replay = pickle.load(f)

    plt.close('all')
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.grid(True, which='major', axis='both')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.spines['left'].set_linewidth(2)
    # ax.spines['bottom'].set_linewidth(2)

    plt.rc('font', family='serif')

    light_blue = np.array([0.529, 0.808, 0.922, 1])
    gold = np.array([1, 0.843, 0, 1])

    color_list_map = plt.cm.get_cmap('Set1', len(replay.keys()))
    color_list = color_list_map(np.linspace(0, 1, len(replay.keys())))

    for index, key in enumerate(replay.keys()):
        env = replay[key]

        date_range = pd.date_range(start=env.sim_starting_date,
                                   end=env.sim_starting_date +
                                   (env.simulation_length - 1) *
                                   datetime.timedelta(
                                       minutes=env.timescale),
                                   freq=f'{env.timescale}min')
        date_range_print = pd.date_range(start=env.sim_starting_date,
                                         end=env.sim_date,
                                         periods=7)

        counter = 1
        dim_x = int(np.ceil(np.sqrt(env.number_of_transformers)))
        dim_y = int(np.ceil(env.number_of_transformers/dim_x))
        for tr in env.transformers:

            plt.subplot(dim_x, dim_y, counter)
            df = pd.DataFrame([], index=date_range)

            if env.config['inflexible_loads']['include']:
                df['inflexible'] = env.tr_inflexible_loads[tr.id, :]
            if env.config['solar_power']['include']:
                df['solar'] = env.tr_solar_power[tr.id, :]

            for cs in tr.cs_ids:
                df[cs] = env.cs_power[cs, :]

            if index == 0:
                # plot the inflexible loads as a fill between
                if env.config['inflexible_loads']['include']:
                    plt.fill_between(df.index,
                                     np.array([0]*len(df.index)),
                                     df['inflexible'],
                                     step='post',
                                     alpha=0.3,
                                     color=light_blue,
                                     linestyle='--',
                                     linewidth=1,
                                     label='Inflexible Loads')

                # plot the solar power as a fill between the inflexible loads and the solar power
                if env.config['solar_power']['include']:
                    plt.fill_between(df.index,
                                     df['inflexible'],
                                     df['solar'] + df['inflexible'],
                                     step='post',
                                     alpha=0.8,
                                     color=gold,
                                     linestyle='--',
                                     linewidth=1,
                                     label='Solar Power')

                if env.config['demand_response']['include']:
                    plt.fill_between(df.index,
                                     np.array([tr.max_power.max()]
                                              * len(df.index)),
                                     tr.max_power,
                                     step='post',
                                     alpha=0.7,
                                     color='r',
                                     hatch='xx',
                                     linestyle='--',
                                     linewidth=2,
                                     label='Demand Response Event')

                # plt.step(df.index,
                #          #  tr.max_power
                #          [-tr.max_power.max()] * len(df.index),
                #          where='post',
                #          color='r',
                #          linestyle='--',
                #          linewidth=1,
                #          alpha=0.7,
                #         #  label='Transf. Limit'
                #          )

                setpoints = env.power_setpoints
                plt.step(df.index,
                         #  tr.max_power
                         setpoints,
                         where='post',
                         color='r',
                         linestyle='--',
                         linewidth=1,
                         alpha=0.7,
                         label='Transf. Limit')

                plt.plot([env.sim_starting_date, env.sim_date],
                         [0, 0], 'black')

            df['total'] = df.sum(axis=1)

            # plot total and use different color and linestyle for each algorithm
            plt.step(df.index, df['total'],
                     color=color_list[index],
                     where='post',
                     linestyle='-',
                     linewidth=1,
                     markevery=5,
                     marker=marker_list[index],
                     label=algorithm_names[index])
            plt.setp(plt.gca().get_lines(), markersize=2)
            plt.setp(plt.gca().get_lines(), markeredgewidth=0.5)

            counter += 1

    plt.ylabel(f'Power (kW)', fontsize=14)
    plt.xlim([env.sim_starting_date, env.sim_date])
    plt.xticks(ticks=date_range_print,
               labels=[
                   f'{d.hour:2d}:{d.minute:02d}' for d in date_range_print],
               #    rotation=45,
               fontsize=14)
    plt.yticks(fontsize=14)
    # put legend under the plot
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
    #            fancybox=True, shadow=True, ncol=3, fontsize=24)

    fig_name = f'{save_path}/Transformer_Aggregated_Power_Prices.png'

    plt.savefig(fig_name, format='png',
                dpi=300, bbox_inches='tight')

    # plt.show()


def plot_comparable_CS_Power(results_path, save_path=None, algorithm_names=None):
    '''
    This function is used to plot the SoC of the EVs in the same plot
    '''

    with open(results_path, 'rb') as f:
        replay = pickle.load(f)

    plt.close('all')
    fig, ax = plt.subplots(figsize=(6, 3.5))
    plt.rc('font', family='serif')
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)

    # ax.spines['left'].set_linewidth(2)
    # ax.spines['bottom'].set_linewidth(2)

    plt.grid(True, which='major', axis='both')
    for index, key in enumerate(replay.keys()):
        env = replay[key]

        date_range = pd.date_range(start=env.sim_starting_date,
                                   end=env.sim_starting_date +
                                   (env.simulation_length - 1) *
                                   datetime.timedelta(
                                       minutes=env.timescale),
                                   freq=f'{env.timescale}min')
        date_range_print = pd.date_range(start=env.sim_starting_date,
                                         end=env.sim_date,
                                         periods=7)

        color_list_map = plt.cm.get_cmap('Set1', len(replay.keys()))
        color_list = color_list_map(np.linspace(0, 1, len(replay.keys())))

        cs_to_plot = 24
        counter = 1
        for cs in env.charging_stations:
            if counter != cs_to_plot:
                counter += 1
                continue

            # plt.subplot(1, 2, counter)
            df = pd.DataFrame([], index=date_range)

            for port in range(cs.n_ports):
                df[port] = env.port_current[port, cs.id, :]

            # multiply df[port] by the voltage to get the power
            df = df * cs.voltage * math.sqrt(cs.phases) / 1000

            # Add another row with one datetime step to make the plot look better
            df.loc[df.index[-1] +
                   datetime.timedelta(minutes=env.timescale)] = df.iloc[-1]

            for port in range(cs.n_ports):
                for i, (t_arr, t_dep) in enumerate(env.port_arrival[f'{cs.id}.{port}']):
                    t_dep = t_dep + 1
                    if t_dep > len(df):
                        t_dep = len(df)
                    # x = df.index[t_arr:t_dep]
                    y = df[port].values.T[t_arr:t_dep]
                    # fill y with 0 before and after to match the length of df
                    y = np.concatenate(
                        [np.zeros(t_arr), y, np.zeros(len(df) - t_dep)])

                    # plt.step(df.index,
                    #          y,
                    #          where='post',
                    #          color=color_list[index],
                    #          marker=marker_list[index],
                    #          alpha=0.8,
                    #          markevery=5,
                    #          label=algorithm_names[index])

                    # do a scatter plot for the markers
                    # plt.scatter(df.index,
                    #             y,
                    #             color=color_list[index],
                    #             marker=marker_list[index],
                    #             alpha=0.8,
                    #             s=10)

                    colorftm = f''
                    # print(f'colorftm: {colorftm}')
                    markerline, stemlines, baseline = plt.stem(df.index,
                                                               y,
                                                               linefmt=colorftm,
                                                               markerfmt=marker_list[index],
                                                               basefmt=' ',
                                                               use_line_collection=True,
                                                               )
                    markerline.set_markerfacecolor(color_list[index])
                    markerline.set_markeredgecolor(color_list[index])
                    # set zorder to plot the markers on top of the lines
                    markerline.set_zorder(10)

                    # set alpha for the markers
                    markerline.set_alpha(0.8)

                    stemlines.set_linewidth(0.2)
                    stemlines.set_color(color_list[index])

                    # change the marker spacing and distance
                    plt.setp(plt.gca().get_lines(), markersize=5)
                    plt.setp(plt.gca().get_lines(), markeredgewidth=0.5)
            # plt.title(f'Charging Station {cs.id + 1}', fontsize=24)

            plt.yticks(fontsize=14)
            plt.ylabel(f'Action [kW]', fontsize=14)
            plt.yticks([-11, -9,  -6, -3, 0, 3, 6, 9,  11],
                       fontsize=14)

            # plt.ylim([0.1, 1.09])
            starting_date = env.sim_starting_date + \
                datetime.timedelta(hours=16)
            ending_date = env.sim_starting_date + datetime.timedelta(hours=55)

            plt.xlim([starting_date, ending_date])
            date_range_print = pd.date_range(start=starting_date,
                                             end=ending_date,
                                             periods=7)

            plt.xticks(ticks=date_range_print,
                       labels=[
                           f'{d.hour:02d}:{d.minute:02d}' for d in date_range_print],
                       #    rotation=45,
                       fontsize=14
                       )

            counter += 1

    # plt.legend(loc='upper center', bbox_to_anchor=(0, -0.15),
    #            fancybox=True, shadow=True, ncol=, fontsize=24)
    # plot line at 11
    # plt.axhline(y=11,
    #             color='yellow',
    #             linestyle='--',
    #             alpha=0.5,
    #             zorder=0,
    #             label='Max Power')
    # plt.axhline(y=-11,
    #             color='yellow',
    #             linestyle='--',
    #             zorder=0,
    #             alpha=0.5)

    plt.grid(True, which='major', axis='both',
             linestyle='--', linewidth=0.5)
    plt.xlabel('Time', fontsize=14)
    plt.tight_layout()

    fig_name = f'{save_path}/CS_Power_single.png'
    plt.savefig(fig_name, format='png',
                dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_path}/CS_Power_single.pdf',
                format='pdf',
                dpi=300,
                bbox_inches='tight')

    # plt.show()


if __name__ == '__main__':

    results_path = './results/eval_25cs_1tr_PST_V2G_ProfixMax_25_5_algos_1_exp_2025_01_08_315329/'
    save_path = './results_analysis/figs/'

    algorithm_names = ['CAFAP', 'GNN-DT', 'DT', 'QT', 'Optimal (Offline)']
    marker_list = ['o', 's', 'D', '^', 'v']
    color_list = ['b', 'g', 'r', 'c', 'm']

    plt.rcParams['font.family'] = 'serif'

    plot_comparable_EV_SoC_single(results_path=results_path + 'plot_results_dict.pkl',
                                  save_path=save_path,
                                  algorithm_names=algorithm_names,
                                  color_list=color_list,
                                  marker_list=marker_list
                                  )

    # exit()
    # plot_comparable_EV_SoC(results_path=results_path + 'plot_results_dict.pkl',
    #                        save_path=save_path,
    #                        algorithm_names=algorithm_names,
    #                        color_list=color_list,
    #                        marker_list=marker_list
    #                        )

    # plot_total_power_V2G(
    #     results_path=results_path + 'plot_results_dict.pkl',
    #     save_path=save_path,
    #     algorithm_names=algorithm_names
    # )

    plot_comparable_CS_Power(
        results_path=results_path + 'plot_results_dict.pkl',
        save_path=save_path,
        algorithm_names=algorithm_names
    )
