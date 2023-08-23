import os

import pandas as pd
import numpy as np

from utils.helper import locate_processed_data


# Class to calculate features from the dataframes / processing pipline is stated below
class FullSessionDataset:
    def __init__(self, name, identifier, project_path):
        os.chdir(r'V:\VirATeC\data\VirATeC_GCN\1_full_sessions')
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

    def calculate_head_direction_amplitude(self, df_sub):
        ampl = np.abs(np.max(df_sub['HeadDirectionYAngle']) - np.min(df_sub['HeadDirectionYAngle']))

        return ampl

    def create_one_transition_dataset(self):
        save_path = self.project_path + '//data//nodes_and_transitions//'

        test = self.df.copy()

        baselines_pupil_diameter = [np.nanmean(self.df['LeftPupilSize']), np.nanmean(self.df['RightPupilSize'])]

        # Remove most variables to speed up the process and rename object variables
        self.df = self.df[['Time', 'GazeTargetObject', 'GazeTargetTimes', 'SituationalComplexity',
                           'ControllerClicked', 'LeftPupilSize', 'RightPupilSize', 'expert_level',
                           '1A', '1B', '2A', '2B', '3A', '3B', '4A', '4B', '5A', '5B', '6A', '6B',
                           '7A', '7B', '8A', '8B', '9A', '9B']].copy()

        self.df['ControllerClicked'] = self.df['ControllerClicked'].astype(bool)
        self.df['SituationalComplexity'] = self.df['SituationalComplexity'].replace('easy', 'E')
        self.df['SituationalComplexity'] = self.df['SituationalComplexity'].replace('complex', 'C')

        self.df = self.df[self.df['GazeTargetObject'] != 'none']

        for start in self.starts:
            # Get ID as number
            identifier = self.ID[2] + self.ID[3] + self.ID[4]
            interval = start    # Save start for interval identifier

            # Select time interval and save target variable (complexity)
            dfsub = self.df[np.logical_and(self.df['Time'] >= start, self.df['Time'] < start + 30)].copy()
            cond = dfsub['expert_level'].iloc[0]

            # Rename and determine AOIs
            dfsub['GazeTargetObject'] = dfsub['GazeTargetObject'].replace('PresentationBoard', 'PB')
            ooi_lst = ['1A', '1B', '2A', '2B', '3A', '3B', '4A', '4B', '5A', '5B', '6A', '6B',
                       '7A', '7B', '8A', '8B', '9A', '9B', 'PB']

            # Reduce dataset to AOI intervals
            dfsub = dfsub[dfsub['GazeTargetObject'].isin(ooi_lst)].reset_index(drop=True)

            # List of transition attributes
            ID_lst = list()
            cond_lst = list()
            interval_lst = list()

            source_lst = list()
            target_lst = list()
            trans_start_lst = list()
            trans_dur_lst = list()
            weight_lst = list()

            AOI_duration_lst = list()
            clicked_lst = list()
            pupil_diameter_lst = list()


            # First source
            source = dfsub['GazeTargetObject'].iloc[0]

            index = 0
            for i in range(1, len(dfsub)):
                index+=1
                if source != dfsub['GazeTargetObject'].iloc[i]:
                    trans_start_lst.append(dfsub['Time'].iloc[i - 1])
                    trans_dur_lst.append(dfsub['Time'].iloc[i] - dfsub['Time'].iloc[i - 1])
                    source_lst.append(source)
                    target_lst.append(dfsub['GazeTargetObject'].iloc[i])
                    weight_lst.append(1)
                    ID_lst.append(identifier)
                    cond_lst.append(cond)
                    interval_lst.append(interval)

                    AOI_duration = dfsub['GazeTargetTimes'].iloc[i-1]
                    AOI_duration_lst.append(AOI_duration)

                    dfs = dfsub.iloc[i-index:i]

                    # Clicks on AOI
                    if any(dfs['ControllerClicked']):
                        clicked_lst.append(1)
                    else:
                        clicked_lst.append(0)

                    # Pupil diameter
                    dfs['LeftPupilBaselineCorrected'] = dfs['LeftPupilSize'].copy()-baselines_pupil_diameter[0]
                    dfs['RightPupilBaselineCorrected'] = dfs['RightPupilSize'].copy()-baselines_pupil_diameter[1]
                    mean_pupil_diameter = np.nanmean(dfs[['LeftPupilBaselineCorrected','RightPupilBaselineCorrected']], axis=1)
                    pupil_diameter_lst.append(np.nanmean(mean_pupil_diameter))

                    source = dfsub['GazeTargetObject'].iloc[i]
                    index -= index #reset index

            placeholder = [np.nan]*len(ID_lst)
            placeholder_node = placeholder + [np.nan]

            #Last object in dataframe
            ID_lst_node = ID_lst + [identifier]
            cond_lst_node = cond_lst + [cond]
            interval_lst_node = interval_lst + [interval]
            trans_start_lst_node = trans_start_lst + [dfsub['Time'].iloc[i]]
            source_lst_node = source_lst + [dfsub['GazeTargetObject'].iloc[i]]
            AOI_duration_lst = AOI_duration_lst + [dfsub['GazeTargetTimes'].iloc[i]]

            dfs = dfsub.iloc[i - index:i]
            # Clicks on AOI
            if any(dfs['ControllerClicked']):
                clicked_lst.append(1)
            else:
                clicked_lst.append(0)

            # Pupil diameter
            dfs['LeftPupilBaselineCorrected'] = dfs['LeftPupilSize'] - baselines_pupil_diameter[0]
            dfs['RightPupilBaselineCorrected'] = dfs['RightPupilSize'] - baselines_pupil_diameter[1]
            mean_pupil_diameter = np.nanmean(dfs[['LeftPupilBaselineCorrected', 'RightPupilBaselineCorrected']], axis=1)
            pupil_diameter_lst.append(np.nanmean(mean_pupil_diameter))

            df_node = pd.DataFrame({'ID': ID_lst_node, 'condition': cond_lst_node, 'start_interval': interval_lst_node,
                                    'start_transition': trans_start_lst_node,'Node': source_lst_node,
                                    'AOI_duration': AOI_duration_lst,
                                    'clicked': clicked_lst,
                                    'pupil_diameter': pupil_diameter_lst,
                                    'click_duration?': placeholder_node,
                                    'feature_location': placeholder_node,
                                    'controller_direction(pointing)_angle': placeholder_node,
                                    'controller_duration_on_AOI': placeholder_node,
                                    'duration_time_until_first fixation': placeholder_node,
                                    })
            df_trans = pd.DataFrame({'ID': ID_lst, 'condition': cond_lst, 'start_interval': interval_lst,
                                     'start_transition': trans_start_lst, 'Source': source_lst,
                                     'Target': target_lst, 'Weight': weight_lst,
                                     'trans_duration': trans_dur_lst,
                                     'trans_amplitude': placeholder,
                                     'trans_velocity': placeholder,
                                     'head_rotation': placeholder,
                                     })

            df_trans.to_csv(save_path + 'ID' + str(identifier) + '_' + str(interval) + '_' + str(cond) + '_trans.csv', index=False)
            df_node.to_csv(save_path + 'ID' + str(identifier) + '_' + str(interval) + '_' + str(cond) + '_node.csv',
                            index=False)

def create_all_transition_datasets():
    project_path = os.path.abspath(os.getcwd())
    data_lst = locate_processed_data()

    print('Create Transition Datasets:')
    for i in range(len(data_lst)):
        name = data_lst['name'].iloc[i]
        identifier = data_lst['ID'].iloc[i]
        print('ID {}'.format(identifier))

        data = FullSessionDataset(name, identifier, project_path)
        # creates transition matrices and saves them into //data//nodes_and_transitions//
        data.create_one_transition_dataset()
