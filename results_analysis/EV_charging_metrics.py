import pandas as pd


data = pd.read_csv(
     './results_analysis/data/eval_50cs_1tr_PST_V2G_ProfixMax_50_PES_5_algos_100_exp_2025_02_02_868691/data.csv')


# group by algotithm and get mean and std
columns = ['Unnamed: 0', 'run', 'Algorithm', 'total_ev_served', 'total_profits',
           'total_energy_charged', 'total_energy_discharged',
           'average_user_satisfaction', 'power_tracker_violation',
           'tracking_error', 'energy_tracking_error', 'energy_user_satisfaction',
           'total_transformer_overload', 'battery_degradation',
           'battery_degradation_calendar', 'battery_degradation_cycling',
           'total_reward']

columns_to_keep = ['Algorithm', 
                   'run',
                   'total_energy_charged',
                   'total_energy_discharged',
                   'average_user_satisfaction',
                   'min_energy_user_satisfaction',
                   'power_tracker_violation',
                #    'energy_tracking_error',
                   'total_profits',
                    'total_reward',
                   'time',
]
data = data[columns_to_keep]

# find the PowerTrackingErrorrMin.total_reward
PowerTrackingErrorrMin = data[data.Algorithm ==
                              "PowerTrackingErrorrMin"]
# find the mean and std of the optimality gap for each algorithm

print(data.head(20))

columns_to_drop = [
    'run',
]

data = data.drop(columns=columns_to_drop)


data_grouped = data.groupby('Algorithm').agg(['mean', 'std'])

# create new columns with the mean and std of the total_energy_charged combined as a string
data_grouped['total_energy_charged'] = data_grouped['total_energy_charged']\
    .apply(lambda x: f"${x['mean']/1000:.1f}$ ±${x['std']/1000:.1f}$", axis=1)
data_grouped['total_energy_discharged'] = data_grouped['total_energy_discharged']\
    .apply(lambda x: f"${x['mean']/1000:.2f}$ ±${x['std']/1000:.2f}$", axis=1)
data_grouped['average_user_satisfaction'] = data_grouped['average_user_satisfaction']\
    .apply(lambda x: f"${x['mean']*100:.1f}$ ±${x['std']*100:.1f}$", axis=1)
data_grouped['min_energy_user_satisfaction'] = data_grouped['min_energy_user_satisfaction']\
    .apply(lambda x: f"${x['mean']:.1f}$", axis=1)
data_grouped['total_profits'] = data_grouped['total_profits']\
    .apply(lambda x: f"${x['mean']:.0f}$ ±${x['std']:.0f}$", axis=1)
data_grouped['power_tracker_violation'] = data_grouped['power_tracker_violation']\
    .apply(lambda x: f"${x['mean']:.1f}$ ±${x['std']:.1f}$", axis=1)
data_grouped['total_reward'] = data_grouped['total_reward']\
       .apply(lambda x: f"${x['mean']/100000:.3f}$ ±${x['std']/100000:.3f}$", axis=1)
data_grouped['time'] = data_grouped['time']\
       .apply(lambda x: f"${x['mean']/300:.3f}$", axis=1)

# rearange rows


# drop the mean and std columns
data_grouped = data_grouped.droplevel(1, axis=1)
# print the results
# drop duplicate columns
data_grouped = data_grouped.loc[:, ~data_grouped.columns.duplicated()]
# rename columns
data_grouped.columns = ['Energy Charged [MWh]',
                        'Energy Discharged [MWh]',
                        'User Satisfaction [%]',
                        'Min. User Satisfaction [%]',
                        'Power Violation [kW]',
                        'Costs [€]',                        
                        'Reward [-]',
                        'Step time [sec/step]',                        
                        ]

print(data_grouped)

# rename algorithm names with shorter names
data_grouped.index = data_grouped.index.str.replace(
    'ChargeAsFastAsPossible', 'CAFAP')
data_grouped.index = data_grouped.index.str.replace(
    'ppo', 'PPO')
# data_grouped.index = data_grouped.index.str.replace(
#     'QT', 'Q-DT')
data_grouped.index = data_grouped.index.str.replace('RoundRobin_GF', 'BaU')
data_grouped.index = data_grouped.index.str.replace(
    'mo_PST_V2GProfitMaxOracleGB', 'Optimal (Offline)')
data_grouped.index = data_grouped.index.str.replace('GNN_act_emb_DT', 'GNN-DT')

print(f'algorithm names: {data_grouped.index}')


# change order of rows
data_grouped = data_grouped.reindex(['CAFAP',
                                     'BaU',
                                     'PPO',
                                     'GNN-DT',
                                     'Optimal (Offline)'
                                     ])


# rename PowerTrackingErrorrMin to Optimal
# print(data_grouped)
print(data_grouped.to_latex())




# \begin{tabular}{lllllllll}
# \toprule
# {} & Energy Charged [MWh] & Energy Discharged [MWh] & User Satisfaction [\%] & Min. User Satisfaction [\%] & Power Violation [kW] &      Costs [€] &         Reward [-] & Step time [sec/step] \\
# Algorithm         &                      &                         &                       &                            &                      &                &                    &                      \\
# \midrule
# CAFAP             &         $2.2$ ±$0.2$ &          $0.00$ ±$0.00$ &        $100.0$ ±$0.0$ &                    $100.0$ &    $3447.0$ ±$447.2$ &  $-462$ ±$252$ &  $-4.581$ ±$0.489$ &              $0.001$ \\
# BaU               &         $2.0$ ±$0.2$ &          $0.00$ ±$0.00$ &         $98.9$ ±$0.4$ &                     $59.5$ &         $9.5$ ±$6.5$ &  $-382$ ±$215$ &  $-0.961$ ±$0.080$ &              $0.001$ \\
# PPO               &         $1.3$ ±$0.1$ &          $0.07$ ±$0.02$ &         $85.6$ ±$1.8$ &                     $16.2$ &     $168.9$ ±$140.5$ &  $-237$ ±$130$ &  $-2.049$ ±$0.311$ &              $0.002$ \\
# GNN-DT            &         $1.5$ ±$0.2$ &          $0.28$ ±$0.04$ &         $97.5$ ±$0.5$ &                     $70.1$ &       $64.2$ ±$67.1$ &  $-227$ ±$131$ &  $-0.096$ ±$0.073$ &              $0.012$ \\
# Optimal (Offline) &         $3.1$ ±$0.3$ &          $1.77$ ±$0.29$ &         $99.0$ ±$0.2$ &                     $76.1$ &       $10.9$ ±$14.7$ &  $-205$ ±$126$ &  $-0.032$ ±$0.025$ &              $0.025$ \\
# \bottomrule
# \end{tabular}






# % \usepackage{tabularray}
# \begin{table}
# \centering
# \captionsetup{labelformat=empty}
# \caption{Analysis of the Overall Reward into its Three Components}
# \label{tab:reward_breakdown}
# \begin{tblr}{
#   cells = {c,t},
#   vline{2} = {2-7}{0.05em},
#   hline{1,8} = {-}{0.08em},
#   hline{2} = {-}{0.05em},
# }
# Algorithm & {Energy Charged\\~[MWh]} & {Energy Discharged\\~[MWh]} & {User Satisfaction\\~[\%]} & {Power Violation\\~[kW]} & Costs~[€] & Reward [10-5] & {Exec. Time\\~[sec/step]}\\
# CAFAP & $1.3$ ±$0.2$ & $0.00$ ±$0.00$ & $100.0$ ±$0.0$ & $1289.2$ ±$261.8$ & $-277$ ±$165$ & $-1.974$ ±$0.283$ & $0.001$\\
# BaU & $1.3$ ±$0.2$ & $0.00$ ±$0.00$ & $99.9$ ±$0.2$ & $10.5$ ±$9.4$ & $-255$ ±$156$ & $-0.679$ ±$0.067$ & $0.001$\\
# DT & $0.9$ ±$0.1$ & $0.03$ ±$0.01$ & $94.4$ ±$1.6$ & $58.7$ ±$28.3$ & $-173$ ±$104$ & $-0.462$ ±$0.093$ & $0.006$\\
# Q-DT & $1.0$ ±$0.1$ & $0.00$ ±$0.00$ & $93.6$ ±$2.1$ & $20.1$ ±$21.4$ & $-187$ ±$113$ & $-0.665$ ±$0.135$ & $0.010$\\
# GNN-DT & $0.9$ ±$0.1$ & $0.19$ ±$0.03$ & $99.3$ ±$0.2$ & $21.7$ ±$22.8$ & $-142$ ±$89$ & $-0.027$ ±$0.023$ & $0.023$\\
# Optimal (Offline) & $1.9$ ±$0.2$ & $1.08$ ±$0.19$ & $99.1$ ±$0.2$ & $2.0$ ±$4.6$ & $-119$ ±$84$ & $-0.020$ ±$0.015$ & -
# \end{tblr}
# \end{table}