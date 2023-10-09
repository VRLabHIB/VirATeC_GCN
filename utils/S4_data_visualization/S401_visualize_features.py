import os

import numpy as np
import pandas as pd
from pingouin import ttest

from utils.helper import locate_node_data
from utils.helper import locate_transition_data


def create_statistic_dataframes(trans_lst, node_lst, project_path):
    data_path = project_path + '\\data\\statistics\\'
    for i in range(len(trans_lst)):
        trans_name = trans_lst['name'].iloc[i]
        # print(trans_name)
        node_name = node_lst['name'].iloc[i]
        # print(node_name)
        identifier = trans_lst['ID'].iloc[i]
        # print('ID {}'.format(identifier))

        if i == 0:
            trans_name = trans_lst['name'].iloc[i]
            node_name = node_lst['name'].iloc[i]

            dft = pd.read_csv(trans_name, low_memory=False)
            dft = dft.rename(columns={'Weight': 'weight'})
            dfn = pd.read_csv(node_name, low_memory=False)

            dfc = pd.concat([dft, dfn.iloc[:, 6:]], axis=1)
            dfc = dfc.drop(columns=['start_transition', 'Source', 'Target', 'weight'])
            columns = list(dft.columns) + list(dfn.columns[6:])

            df_max = pd.DataFrame(columns=columns)
            df_max = df_max.drop(columns=['start_transition', 'Source', 'Target', 'weight'])
            df_max.loc[len(df_max)] = np.concatenate((dfc.iloc[0, :4].values, np.max(dfc.iloc[:, 4:], axis=0).values), axis=None)

            df_min = pd.DataFrame(columns=columns)
            df_min = df_min.drop(columns=['start_transition', 'Source', 'Target', 'weight'])
            df_min.loc[len(df_min)] = np.concatenate((dfc.iloc[0, :4].values, np.min(dfc.iloc[:, 4:], axis=0).values), axis=None)

            df_mean = pd.DataFrame(columns=columns)
            df_mean = df_mean.drop(columns=['start_transition', 'Source', 'Target', 'weight'])
            df_mean.loc[len(df_mean)] = np.nanmean(dfc, axis=0)

            df_std = pd.DataFrame(columns=columns)
            df_std = df_std.drop(columns=['start_transition', 'Source', 'Target', 'weight'])
            df_std.loc[len(df_std)] = np.concatenate((dfc.iloc[0, :4].values, np.nanstd(dfc.iloc[:, 4:], axis=0)), axis=None)

        if i != 0:
            dft = pd.read_csv(trans_name, low_memory=False)
            dft = dft.rename(columns={'Weight': 'weight'})
            dfn = pd.read_csv(node_name, low_memory=False)

            if len(dft) != 0:
                dfc = pd.concat([dft, dfn.iloc[:, 6:]], axis=1)
                dfc = dfc.drop(columns=['start_transition', 'Source', 'Target', 'weight'])

                df_max.loc[len(df_max)] = np.concatenate((dfc.iloc[0, :4].values, np.max(dfc.iloc[:, 4:], axis=0).values), axis=None)
                df_min.loc[len(df_min)] = np.concatenate((dfc.iloc[0, :4].values, np.min(dfc.iloc[:, 4:], axis=0).values), axis=None)
                df_mean.loc[len(df_mean)] = np.nanmean(dfc, axis=0)
                df_std.loc[len(df_std)] = np.concatenate((dfc.iloc[0, :4].values, np.nanstd(dfc.iloc[:, 4:], axis=0)), axis=None)

    df_max.to_csv(data_path + 'max_val_per_interval.csv', index=False)
    df_min.to_csv(data_path + 'min_val_per_interval.csv', index=False)
    df_mean.to_csv(data_path + 'mean_val_per_interval.csv', index=False)
    df_std.to_csv(data_path + 'std_val_per_interval.csv', index=False)


def load_statistic_dataframes(project_path):
    data_path = project_path + '\\data\\statistics\\'

    df_max = pd.read_csv(data_path + 'max_val_per_interval.csv')
    df_min = pd.read_csv(data_path + 'min_val_per_interval.csv')
    df_mean = pd.read_csv(data_path + 'mean_val_per_interval.csv')
    df_std = pd.read_csv(data_path + 'std_val_per_interval.csv')

    return df_max, df_min, df_mean, df_std


if __name__ == '__main__':
    project_path = os.path.abspath(os.getcwd()).rsplit('\\', 2)[0]
    save_path = project_path + '\\data\\nodes_and_transitions\\'

    trans_lst = locate_transition_data(project_path)
    node_lst = locate_node_data(project_path)

    create_statistic_dataframes(trans_lst, node_lst, project_path)

    df_max, df_min, df_mean, df_std = load_statistic_dataframes(project_path)

    for col in range(4, len(df_mean.columns)):
        #zero = df_mean[df_mean['Complexity'] == 0][df_mean.columns[col]].values
        #one = df_mean[df_mean['Complexity'] == 1][df_mean.columns[col]].values

        # zero = df_std[df_std['Complexity'] == 0][df_mean.columns[col]].values
        # one = df_std[df_std['Complexity'] == 1][df_mean.columns[col]].values

        zero = df_max[df_max['Complexity'] == 0][df_mean.columns[col]].values
        one = df_max[df_max['Complexity'] == 1][df_mean.columns[col]].values

        if col == 4:
            stats = ttest(zero, one, paired=False).round(2)
            stats.index = [df_mean.columns[col]]

        if col != 4:
            stat = ttest(zero, one, paired=False).round(2)
            stat.index = [df_mean.columns[col]]

            stats = pd.concat([stats, stat], axis=0)

        if df_max.columns[col] == 'pupil_diameter':
            s = df_max.groupby('Complexity')['pupil_diameter'].describe()
    print(' ')
