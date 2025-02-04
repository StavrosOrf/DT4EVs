import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import math
import seaborn as sns

def unwrap_env(env):
    while hasattr(env, "env"):
        env = env.env  # Go one level deeper
    return env

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
        print(f'env: {env}, type: {str(type(env))}')
        if str(type(env)) == "<class 'gymnasium.wrappers.common.OrderEnforcing'>":
            print('Order enforcing')
            env = unwrap_env(env) 

        # color_list_map = plt.cm.get_cmap('Set1', len(replay.keys()))
        # color_list = color_list_map(np.linspace(0, 1, len(replay.keys())))
        color_list = [
            sns.color_palette()[3],
            sns.color_palette()[2],
            sns.color_palette()[0],
            sns.color_palette()[1],
            sns.color_palette()[7],
        ]

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

            if counter == 25 and index == 0:
                plt.axhline(y=0.8,
                            color=sns.color_palette()[4],
                            linestyle='--',
                            alpha=0.5,
                            label='Desired SoC')
            else:
                plt.axhline(y=0.8,
                            color=sns.color_palette()[4],
                            linestyle='--',
                            alpha=0.5)

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
                                 label=algorithm_names[index],
                                 zorder=10
                                 )
                    else:
                        plt.step(df.index,
                                 y,
                                 where='post',
                                 color=color_list[index],
                                 marker=marker_list[index],
                                 zorder=10
                                 )

                    # change the marker spacing and distance
                    plt.setp(plt.gca().get_lines(), markersize=0.5)
                    plt.setp(plt.gca().get_lines(), markeredgewidth=0.3)

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
               fancybox=True, shadow=True, ncol=6, fontsize=14)

    plt.grid(True, which='minor', axis='both')
    plt.tight_layout()
    plt.subplots_adjust(
        wspace=0.117,    # Width space between subplots
        hspace=0.279     # Height space between subplots
    )

    fig_name = f'{save_path}EV_Energy_Level.png'
    plt.savefig(fig_name, format='png',
                dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_path}EV_Energy_Level.pdf',
                format='pdf',
                # dpi=300,
                bbox_inches='tight')

    # plt.show()


def plot_actual_power_vs_setpoint(results_path,
                                  save_path=None,
                                  algorithm_names=None):
    '''
    This function is used to plot the actual power vs the setpoint power.
    It plots the behavior of each algorithm in subplots vertically.
    '''

    with open(results_path, 'rb') as f:
        replay = pickle.load(f)

    plt.close('all')
    plt.figure(figsize=(6, 4))
    plt.rc('font', family='serif')

    for index, key in enumerate(replay.keys()):
        if index == 0:
            continue
        env = replay[key]
        print(f'env: {env}, type: {str(type(env))}')
        if str(type(env)) == "<class 'gymnasium.wrappers.common.OrderEnforcing'>":
            print('Order enforcing')
            env = unwrap_env(env) 

        date_range = pd.date_range(start=env.sim_starting_date,
                                   end=env.sim_starting_date +
                                   (env.simulation_length - 1) *
                                   datetime.timedelta(
                                       minutes=env.timescale),
                                   freq=f'{env.timescale}min')
        
        date_range_print = pd.date_range(start=env.sim_starting_date,
                                         end=env.sim_date,
                                         periods=5)

        # plot the actual power vs the setpoint power for each algorithm in subplots
        plt.subplot(2, 2, index)
        plt.grid(True, which='major', axis='both')

        actual_power = env.current_power_usage
        setpoints = env.power_setpoints

        plt.step(date_range, actual_power.T, alpha=0.9, color='#00429d')
        plt.step(date_range, setpoints.T, alpha=1, color='#93003a')
        
        # fill between the actual power and 0
        plt.fill_between(date_range, actual_power.T, 0, step='post', alpha=0.3, color='#00429d')        
        

        plt.axhline(0, color='black', lw=2)
        plt.title(f'{algorithm_names[index]}', fontsize=13)

        plt.yticks(fontsize=14)

        if index == 3 or index == 4:
            plt.xticks(ticks=date_range_print,
                    labels=[
                        f'{d.hour:2d}:{d.minute:02d}' for d in date_range_print],
                    # rotation=45,
                    fontsize=10)        
        else:
            plt.xticks(ticks=date_range_print,
                       labels=[' ' for d in date_range_print])

        
        if index == 1:
            plt.yticks(np.arange(0, 125, 40), 
                       fontsize=12)
        elif index == 2:
            plt.yticks(np.arange(0, 125, 40),
                          labels=' '*4,
                          fontsize=12)
        elif index == 3:
            plt.yticks(np.arange(-40, 125, 40), 
                       fontsize=12)
        else:
            plt.yticks(np.arange(-40, 125, 40),
                          labels=' '*5,
                          fontsize=12)
        
        if index == 1 or index == 3:
            plt.ylabel('Power [kW]', fontsize=11)
        else:
            plt.ylabel('', fontsize=12)

        plt.xlim([env.sim_starting_date, env.sim_date])
        # plt.ylim([0, 120])

    # Put the legend under the plot in a separate axis
    plt.legend(['Aggregated Power', 'Dynamic Power Limit'], loc='upper center',
               bbox_to_anchor=(-0.15,-0.2),
               fancybox=True, shadow=True, ncol=2, fontsize=12)

    plt.subplots_adjust(
        left=0.055,    # Space from the left of the figure
        bottom=0.1,   # Space from the bottom of the figure
        right=0.986,   # Space from the right of the figure
        top=0.955,     # Space from the top of the figure
        hspace=0.4,    # Height space between subplots
        wspace=0.229    # Width space between subplots
    )

    # plt.tight_layout()
    fig_name = f'{save_path}/Actual_vs_Setpoint_Power.png'
    plt.savefig(fig_name, format='png',
                dpi=300,
                bbox_inches='tight',
                )
    plt.savefig(f'{save_path}/Actual_vs_Setpoint_Power.pdf',
                format='pdf',
                # dpi=300,
                bbox_inches='tight',
                )

    plt.show()


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
    plt.axhline(y=80,
                color=sns.color_palette()[4],
                linestyle='--',
                alpha=0.5,
                label='Desired SoC')

    for index, key in enumerate(replay.keys()):
        env = replay[key]
        print(f'env: {env}, type: {str(type(env))}')
        if str(type(env)) == "<class 'gymnasium.wrappers.common.OrderEnforcing'>":
            print('Order enforcing')
            env = unwrap_env(env) 

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

        color_list = [
            sns.color_palette()[3],
            sns.color_palette()[2],
            sns.color_palette()[0],
            sns.color_palette()[1],
            sns.color_palette()[7],

        ]

        cs_to_plot = 18
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
                    y = y * 100

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

            plt.ylabel('SoC [%]', fontsize=14)
            plt.yticks(np.arange(0, 110, 20),
                       fontsize=14)

            # plt.xlabel('Time', fontsize=14)

            plt.ylim([20, 102])
            # plt.xlim([env.sim_starting_date, env.sim_date])
            # plt.xticks(ticks=date_range_print,
            #            labels=[f'{d.hour:2d}:{d.minute:02d}' for d in date_range_print],
            #            rotation=45,
            #            fontsize=14)

            starting_date = env.sim_starting_date + \
                datetime.timedelta(hours=10)
            ending_date = env.sim_starting_date + datetime.timedelta(hours=40)

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
        print(f'env: {env}, type: {str(type(env))}')
        if str(type(env)) == "<class 'gymnasium.wrappers.common.OrderEnforcing'>":
            print('Order enforcing')
            env = unwrap_env(env) 

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
        print(f'env: {env}, type: {str(type(env))}')
        if str(type(env)) == "<class 'gymnasium.wrappers.common.OrderEnforcing'>":
            print('Order enforcing')
            env = unwrap_env(env) 

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

        color_list = [
            sns.color_palette()[3],
            sns.color_palette()[2],
            sns.color_palette()[0],
            sns.color_palette()[1],
            sns.color_palette()[7],

        ]

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


def plot_comparable_CS_Power_scatterplot(results_path, save_path=None, algorithm_names=None):

    with open(results_path, 'rb') as f:
        replay = pickle.load(f)

    plt.close('all')
    fig, ax = plt.subplots(figsize=(6, 3))
    plt.rc('font', family='serif')
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)

    # ax.spines['left'].set_linewidth(2)
    # ax.spines['bottom'].set_linewidth(2)

    plt.grid(True, which='major', axis='both')
    

    all_data = pd.DataFrame([], index=[])
    for index, key in enumerate(replay.keys()):
        env = replay[key]
        print(f'env: {env}, type: {str(type(env))}')
        if str(type(env)) == "<class 'gymnasium.wrappers.common.OrderEnforcing'>":
            print('Order enforcing')
            env = unwrap_env(env)        

        color_list_map = plt.cm.get_cmap('Set1', len(replay.keys()))
        color_list = color_list_map(np.linspace(0, 1, len(replay.keys())))

        color_list = [
            sns.color_palette()[3],
            sns.color_palette()[2],
            sns.color_palette()[0],
            sns.color_palette()[1],
            sns.color_palette()[7],

        ]

        cs_to_plot = 24
        actions = []
        counter = 1
        for cs in env.charging_stations:

            for port in range(cs.n_ports):
                for i, (t_arr, t_dep) in enumerate(env.port_arrival[f'{cs.id}.{port}']):

                    actions.append(env.port_current[port, cs.id, t_arr:t_dep] *
                                   cs.voltage * math.sqrt(cs.phases) / 1000)
                    act = (env.port_current[port, cs.id, t_arr:t_dep] *
                                   cs.voltage * math.sqrt(cs.phases) / 1000)
                    
                    for a in act:                                                
                        df = pd.DataFrame({
                            'Action': [a],
                            'Algorithm': algorithm_names[index]
                        })
                        all_data = pd.concat([all_data, df])            
                        
    all_data = all_data.reset_index(drop=True)
    
    print(f'unique algorithms: {all_data["Algorithm"].unique()}')
    
    sns.stripplot(data=all_data,
                  x='Algorithm',
                  y='Action',
                  hue='Algorithm',                  
                  jitter=0.35,
                  palette=color_list,
                  size=2)
    
    plt.xlabel('', fontsize=14)
    plt.ylabel(f'Action [kW]', fontsize=14)
    plt.yticks([-11, -9,  -6, -3, 0, 3, 6, 9,  11],
                fontsize=14)
    #plot a line at y=0
    plt.axhline(y=0,
                color='black',
                # linestyle='--',
                linewidth=0.5,
                alpha=0.7)
    
    plt.grid(True, which='major', axis='both',
             linestyle='--', linewidth=0.5)
    

    fig_name = f'{save_path}/CS_Power_scatter.png'
    plt.savefig(fig_name, format='png',
                dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_path}/CS_Power_scatter.pdf',
                format='pdf',
                dpi=300,
                bbox_inches='tight')


if __name__ == '__main__':

    # results_path = './results/eval_50cs_1tr_PST_V2G_ProfixMax_50_PES_5_algos_100_exp_2025_02_02_868691/'
    results_path = './results/eval_50cs_1tr_PST_V2G_ProfixMax_50_PES_5_algos_1_exp_2025_02_03_727655/'
    # results_path = './results/eval_50cs_1tr_PST_V2G_ProfixMax_50_PES_5_algos_1_exp_2025_02_04_757687/'
    save_path = './results_analysis/figs/'

    algorithm_names = ['CAFAP','BaU','PPO','GNN + LLM', 'Optimal (Offline)']
    algorithm_names = ['CAFAP','BaU','GNN + LLM','PPO', 'Optimal (Offline)']
    marker_list = ['o', 's', 'D', '^', 'v']
    color_list = ['b', 'g', 'r', 'c', 'm']

    plt.rcParams['font.family'] = 'serif'

    plot_comparable_EV_SoC_single(results_path=results_path + 'plot_results_dict.pkl',
                                  save_path=save_path,
                                  algorithm_names=algorithm_names,
                                  color_list=color_list,
                                  marker_list=marker_list
                                  )
    # plot_comparable_CS_Power(
    #     results_path=results_path + 'plot_results_dict.pkl',
    #     save_path=save_path,
    #     algorithm_names=algorithm_names
    # )

    # plot_comparable_CS_Power_scatterplot(
    #     results_path=results_path + 'plot_results_dict.pkl',
    #     save_path=save_path,
    #     algorithm_names=algorithm_names
    # )

    # exit()
    # plot_comparable_EV_SoC(results_path=results_path + 'plot_results_dict.pkl',
    #                        save_path=save_path,
    #                        algorithm_names=algorithm_names,
    #                        color_list=color_list,
    #                        marker_list=marker_list
    #                        )

    # plot_actual_power_vs_setpoint(
    #     results_path=results_path + 'plot_results_dict.pkl',
    #     save_path=save_path,
    #     algorithm_names=algorithm_names
    # )

    # plot_total_power_V2G(
    #     results_path=results_path + 'plot_results_dict.pkl',
    #     save_path=save_path,
    #     algorithm_names=algorithm_names
    # )
