import os

import pandas as pd
import numpy as np

from utils.helper import locate_processed_data


# Class to calculate features from the dataframes / processing pipline is stated below
class FullSessionDataset:
    def __init__(self, name, identifier, project_path):
        os.chdir(r'V:\VirATeC\data\VirATeC_NSR\1_full_sessions')
        self.df = pd.read_csv(name, low_memory=False)
        self.ID = identifier
        self.project_path = project_path

        self.starts = np.arange(0, 601, 30)
        self.ends = np.arange(30, 631, 30)
        self.df_lst = []

    def get_data(self):
        return self.df

    def get_matrices(self):
        return self.df_lst

    def get_ID(self):
        return self.ID

    def create_one_transition_dataset(self):
        # Remove most variables to speed up the process and rename object variables
        self.df = self.df[['Time', 'GazeTargetObject', 'GazeTargetTimes', 'SituationalComplexity',
                           '1A', '1B', '2A', '2B', '3A', '3B', '4A', '4B', '5A', '5B', '6A', '6B',
                           '7A', '7B', '8A', '8B', '9A', '9B']].copy()

        self.df['SituationalComplexity'] = self.df['SituationalComplexity'].replace('easy', 'E')
        self.df['SituationalComplexity'] = self.df['SituationalComplexity'].replace('complex', 'C')
        save_path = self.project_path + '//data//transitions//'

        self.df = self.df[self.df['GazeTargetObject'] != 'none']

        for start in self.starts:
            # Get ID as number
            identifier = self.ID[2] + self.ID[3] + self.ID[4]
            interval = start    # Save start for interval identifier

            # Select time interval and save target variable (complexity)
            dfsub = self.df[np.logical_and(self.df['Time'] >= start, self.df['Time'] < start + 30)].copy()
            cond = dfsub['SituationalComplexity'].iloc[0]

            # Rename and determine AOIs
            dfsub['GazeTargetObject'] = dfsub['GazeTargetObject'].replace('PresentationBoard', 'PB')
            ooi_lst = ['1A', '1B', '2A', '2B', '3A', '3B', '4A', '4B', '5A', '5B', '6A', '6B',
                       '7A', '7B', '8A', '8B', '9A', '9B', 'PB']

            # Reduce dataset to AOI intervals
            dfsub = dfsub[dfsub['GazeTargetObject'].isin(ooi_lst)].reset_index(drop=True)

            # dfsub['PositionLeftRight'] = dfsub['GazeTargetObject'].replace({'1A':'L', '1B':'L', '4A':'L', '4B':'L',
            # '7A':'L', '7B':'L', '2A':'M', '2B':'M', '5A':'M', '5B':'M', '8A':'M', '8B':'M', '3A':'R', '3B':'R',
            # '6A':'R', '6B':'R', '9A':'R', '9B':'R'})
            # dfsub['PositionFrontBack'] = dfsub['GazeTargetObject'].replace({'1A':'L', '1B':'L', '4A':'L', '4B':'L',
            # '7A':'L', '7B':'L', '2A':'M', '2B':'M', '5A':'M', '5B':'M', '8A':'M', '8B':'M', '3A':'R', '3B':'R',
            # '6A':'R', '6B':'R', '9A':'R', '9B':'R'})

            # List of transition attributes
            ID_lst = list()
            cond_lst = list()
            interval_lst = list()

            source_lst = list()
            target_lst = list()
            trans_start_lst = list()
            trans_dur_lst = list()
            weight_lst = list()

            # First source
            source = dfsub['GazeTargetObject'].iloc[0]

            for i in range(1, len(dfsub)):
                if source != dfsub['GazeTargetObject'].iloc[i]:
                    trans_start_lst.append(dfsub['Time'].iloc[i - 1])
                    trans_dur_lst.append(dfsub['Time'].iloc[i] - dfsub['Time'].iloc[i - 1])
                    source_lst.append(source)
                    target_lst.append(dfsub['GazeTargetObject'].iloc[i])
                    weight_lst.append(1)
                    ID_lst.append(identifier)
                    cond_lst.append(cond)
                    interval_lst.append(interval)

                    source = dfsub['GazeTargetObject'].iloc[i]

            placeholder = [np.nan]*len(ID_lst)

            # TODO: create dataframe with node features: AOI Pupil etc.

            df_trans = pd.DataFrame({'ID': ID_lst, 'condition': cond_lst, 'start_interval': interval_lst,
                                     'start_transition': trans_start_lst, 'Source': source_lst,
                                     'Target': target_lst, 'Weight': weight_lst,
                                     'trans_duration': trans_dur_lst,
                                     'trans_amplitude': placeholder,
                                     'head_rotation': placeholder,
                                     'table_switched': placeholder,
                                     'row_switched': placeholder,
                                     'AOI_duration_Source': placeholder,
                                     'pupil Diameter_Source': placeholder,
                                     'active_disruption_Source': placeholder,
                                     'passive_disruption_Source': placeholder,
                                     'clicked_Source':placeholder,
                                     'AOI_duration_Target': placeholder,
                                     'pupil Diameter_Target': placeholder,
                                     'active_disruption_Target': placeholder,
                                     'passive_disruptionTarget': placeholder,
                                     'clicked_Target': placeholder,
                                     })

            df_trans.to_csv(save_path + 'ID' + str(identifier) + '_' + str(interval) + '_' + str(cond) + '.csv', index=False)


def create_all_transition_datasets():
    project_path = os.path.abspath(os.getcwd())
    data_lst = locate_processed_data()

    print('Create Transition Datasets:')
    for i in range(len(data_lst)):
        name = data_lst['name'].iloc[i]
        identifier = data_lst['ID'].iloc[i]
        print('ID {}'.format(identifier))

        data = FullSessionDataset(name, identifier, project_path)
        # creates transition matrices and saves them into //data//transitions//
        data.create_one_transition_dataset()
