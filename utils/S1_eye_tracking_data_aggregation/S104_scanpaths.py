import os

import pandas as pd
import numpy as np

from utils.S1_eye_tracking_data_aggregation.helper import locate_processed_data


# Class to calculate features from the dataframes / processing pipline is stated below
class ScanPathDataset:
    def __init__(self, name, ID, project_path):
        os.chdir(r'V:\VirATeC\data\VirATeC_NSR\1_full_sessions')
        self.df = pd.read_csv(name, low_memory=False)
        self.ID = ID
        self.project_path = project_path

        starts = np.arange(0, 601, 30)
        ends = np.arange(30, 631, 30)
        self.df_lst = []

    def get_data(self):
        return self.df

    def get_matrices(self):
        return self.df_lst

    def get_ID(self):
        return self.ID

    def create_matrix(self):
        # Remove most variables to speed up the process and rename object variables
        self.df = self.df[['Time', 'GazeTargetObject', 'GazeTargetTimes', 'SituationalComplexity',
                           '1A', '1B', '2A', '2B', '3A', '3B', '4A', '4B', '5A', '5B', '6A', '6B',
                           '7A', '7B', '8A', '8B', '9A', '9B']].copy()

        self.df['SituationalComplexity'] = self.df['SituationalComplexity'].replace('easy', 'E')
        self.df['SituationalComplexity'] = self.df['SituationalComplexity'].replace('complex', 'C')
        save_path = self.project_path + '//data//transitions//'

        self.df = self.df[self.df['GazeTargetObject'] != 'none']

        intervals = [30, 120, 210, 300, 420, 510]

        for start in intervals:
            dfsub = self.df[np.logical_and(self.df['Time'] >= start, self.df['Time'] < start + 90)]

            ooi_lst = ['1A', '1B', '2A', '2B', '3A', '3B', '4A', '4B', '5A', '5B', '6A', '6B',
                       '7A', '7B', '8A', '8B', '9A', '9B', 'PresentationBoard']

            bool_lst = dfsub['GazeTargetObject'].isin(ooi_lst)
            dfsub = dfsub[bool_lst]
            dfsub = dfsub.reset_index(drop=True)
            # dfsub['GazeTargetObject'] = dfsub['GazeTargetObject'].replace({'1A':'L', '1B':'L', '4A':'L', '4B':'L',
            # '7A':'L', '7B':'L', '2A':'M', '2B':'M', '5A':'M', '5B':'M', '8A':'M', '8B':'M', '3A':'R', '3B':'R',
            # '6A':'R', '6B':'R', '9A':'R', '9B':'R'})

            dfsub['GazeTargetObject'] = dfsub['GazeTargetObject'].replace({'PresentationBoard': 'PB'})
            ID = self.ID[2] + self.ID[3] + self.ID[4]
            cond = dfsub['SituationalComplexity'].iloc[0]
            interval = start

            ID_lst = list()
            cond_lst = list()
            interval_lst = list()

            source_lst = list()
            target_lst = list()
            time_lst = list()
            trans_time_lst = list()
            weight_lst = list()

            source = dfsub['GazeTargetObject'].iloc[0]
            t = dfsub['Time'].iloc[0]

            for i in range(1, len(dfsub)):
                if source != dfsub['GazeTargetObject'].iloc[i]:
                    time_lst.append(dfsub['Time'].iloc[i - 1])
                    trans_time_lst.append(dfsub['Time'].iloc[i] - dfsub['Time'].iloc[i - 1])
                    source_lst.append(source)
                    target_lst.append(dfsub['GazeTargetObject'].iloc[i])
                    weight_lst.append(1)
                    ID_lst.append(ID)
                    cond_lst.append(cond)
                    interval_lst.append(interval)

                    source = dfsub['GazeTargetObject'].iloc[i]

            df_trans = pd.DataFrame({'ID': ID_lst, 'condition': cond_lst, 'start_interval': interval_lst,
                                     'time_point': time_lst, 'trans_dur': trans_time_lst,
                                     'Source': source_lst,
                                     'Target': target_lst, 'Weight': weight_lst})

            df_trans.to_csv(save_path + str(cond) + '_' + str(ID) + '_' + str(interval) + '.csv', index=False)


def create_scanpaths():
    project_path = os.path.abspath(os.getcwd())
    data_lst = locate_processed_data()

    df_features = pd.DataFrame()
    print('Create Scanpaths')
    for i in range(len(data_lst)):
        name = data_lst['name'].iloc[i]
        ID = data_lst['ID'].iloc[i]
        print('ID {}'.format(ID))

        data = ScanPathDataset(name, ID, project_path)
        # creates transition matrices and saves them into //data//transitions//
        data.create_matrix()
