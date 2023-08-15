import glob
import os

import numpy as np
import pandas as pd


def locate_raw_data():
    os.chdir(r'V:\VirATeC\data\VirATeC_NSR\0_raw_data')
    lst = glob.glob("*Main.csv")

    df_lst = pd.DataFrame({'name': lst})
    df_lst['ID'] = df_lst['name'].str.split('_', expand=True).iloc[:, 0]

    return df_lst


def locate_processed_data():
    os.chdir(r'V:\VirATeC\data\VirATeC_NSR\1_full_sessions')
    lst = glob.glob("*.csv")

    df_lst = pd.DataFrame({'name': lst})
    df_lst['ID'] = df_lst['name'].str.split('.', expand=True).iloc[:, 0]

    return df_lst


def calculate_time_diff(df):
    """

    Parameters
    ----------
    df : pandas dataframe
        raw, cleaned dataframe.
    time : string
        variable that stores the time changes of an experiment [sec].

    Returns
    -------
    time_diff : list
        conveys time difference between following steps of an experiment.

    """

    # create list with start time of the experiment
    time_diff = np.array([0])

    # create two vectors
    t0 = np.array(df['Time'][:-1].tolist())
    t1 = np.array(df['Time'][1:].tolist())

    # vectors subtraction
    diff = np.subtract(t1, t0)

    time_diff = np.append(time_diff, diff)

    return time_diff


def calculate_object_stats(df):
    df_stats = pd.DataFrame(df['GazeTargetObject'].value_counts() / len(df) * 100)
    df_stats = df_stats.rename(columns={'GazeTargetObject': 'PercentLookedAt'})
    df_stats.insert(0, 'GazeTargetObject', df_stats.index)
    df_stats = df_stats.reset_index(drop=True)

    lst = ['PresentationBoard', 'none']
    line = pd.DataFrame({"GazeTargetObject": 'Child_total',
                         "PercentLookedAt": len(df[~df['GazeTargetObject'].isin(lst)]) / len(df) * 100}, index=[3])
    df_stats = pd.concat([df_stats.iloc[:2], line, df_stats.iloc[2:]]).reset_index(drop=True)

    return df_stats
