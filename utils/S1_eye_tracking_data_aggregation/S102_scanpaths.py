import os

import pandas as pd
import numpy as np

from utils.helper import locate_processed_data


# Class to calculate features from the dataframes / processing pipline is stated below
def calculate_head_direction_amplitude(df_sub):
    ampl = np.abs(np.max(df_sub['HeadDirectionYAngle']) - np.min(df_sub['HeadDirectionYAngle']))

    return ampl


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

    def create_one_transition_dataset(self):
        save_path = self.project_path + '//data//nodes_and_transitions//'

        baselines_pupil_diameter = [np.nanmean(self.df['LeftPupilSize']), np.nanmean(self.df['RightPupilSize'])]

        # Remove most variables to speed up the process and rename object variables
        self.df = self.df[['Time', 'GazeTargetObject', 'GazeTargetObjectTimes', 'SituationalComplexity',
                           'ControllerClicked', 'LeftPupilSize', 'RightPupilSize', 'ExpertLevel', 'HeadDirectionYAngle',
                           '1A', '1B', '2A', '2B', '3A', '3B', '4A', '4B', '5A', '5B', '6A', '6B',
                           '7A', '7B', '8A', '8B', '9A', '9B']].copy()

        self.df['ControllerClicked'] = self.df['ControllerClicked'].astype(bool)
        self.df['SituationalComplexity'] = self.df['SituationalComplexity'].replace('easy', 0)
        self.df['SituationalComplexity'] = self.df['SituationalComplexity'].replace('complex', 1)

        self.df = self.df[self.df['GazeTargetObject'] != 'none']

        for start in self.starts:
            # Get ID as number
            identifier = self.ID[2] + self.ID[3] + self.ID[4]
            interval = start  # Save start for interval identifier

            # Select time interval and save target variable (complexity)
            dfsub = self.df[np.logical_and(self.df['Time'] >= start, self.df['Time'] < start + 30)].copy()
            expertise = dfsub['ExpertLevel'].iloc[0]
            complexity = dfsub['SituationalComplexity'].iloc[0]

            # Rename and determine AOIs
            dfsub['GazeTargetObject'] = dfsub['GazeTargetObject'].replace('PresentationBoard', 'PB')
            ooi_lst = ['1A', '1B', '2A', '2B', '3A', '3B', '4A', '4B', '5A', '5B', '6A', '6B',
                       '7A', '7B', '8A', '8B', '9A', '9B', 'PB']

            # Reduce dataset to AOI intervals
            dfsub = dfsub[dfsub['GazeTargetObject'].isin(ooi_lst)].reset_index(drop=True)

            # List of transition attributes
            id_lst_edge = list()
            expert_lst_edge = list()
            complex_lst_edge = list()
            interval_lst_edge = list()

            source_lst_edge = list()
            target_lst_edge = list()
            trans_start_lst_edge = list()
            trans_dur_lst_edge = list()
            weight_lst_edge = list()

            head_amplitude_lst_edge = list()

            # List of node attributes
            id_lst_node = list()
            expert_lst_node = list()
            complex_lst_node = list()
            interval_lst_node = list()
            duration_start_lst_node = list()
            source_lst_node = list()

            aoi_duration_lst_node = list()
            clicked_lst_node = list()
            pupil_diameter_lst_node = list()
            controller_duration_on_aoi_lst_node = list() #TODO
            distance_to_aoi_lst_node = list() #TODO

            seating_row_aoi_lst_node = list() #TODO
            seating_loc_aoi_lst_node = list() #TODO

            # First source
            source = dfsub['GazeTargetObject'].iloc[0]

            index = 0
            for i in range(1, len(dfsub)):
                index += 1
                if source != dfsub['GazeTargetObject'].iloc[i]:
                    trans_dur = dfsub['Time'].iloc[i] - dfsub['Time'].iloc[i - 1]
                    if trans_dur < 1:
                        trans_start_lst_edge.append(dfsub['Time'].iloc[i - 1])
                        trans_dur_lst_edge.append(trans_dur)
                        source_lst_edge.append(source)
                        target_lst_edge.append(dfsub['GazeTargetObject'].iloc[i])
                        weight_lst_edge.append(1)
                        id_lst_edge.append(identifier)
                        complex_lst_edge.append(complexity)
                        expert_lst_edge.append(expertise)
                        interval_lst_edge.append(interval)
                        head_amplitude_lst_edge.append(np.abs(dfsub['HeadDirectionYAngle'].iloc[i]
                                                              - dfsub['HeadDirectionYAngle'].iloc[i - 1]))

                    #### Select AOI interval ####
                    dfs = dfsub.iloc[i - index:i]

                    id_lst_node.append(identifier)
                    complex_lst_node.append(complexity)
                    expert_lst_node.append(expertise)
                    interval_lst_node.append(interval)
                    duration_start_lst_node.append(dfs['Time'].iloc[0])
                    source_lst_node.append(source)

                    # AOI duration
                    aoi_duration = dfs['GazeTargetObjectTimes'].iloc[0]
                    aoi_duration_lst_node.append(aoi_duration)

                    # Clicks on AOI
                    if any(dfs['ControllerClicked']):
                        clicked_lst_node.append(1)
                    else:
                        clicked_lst_node.append(0)

                    # Pupil diameter
                    df_p = pd.DataFrame(
                        {'LeftPupilBaselineCorrected': dfs['LeftPupilSize'] - baselines_pupil_diameter[0],
                         'RightPupilBaselineCorrected': dfs['RightPupilSize'] - baselines_pupil_diameter[1]})

                    mean_pupil_diameter = np.nanmean(df_p, axis=1)
                    pupil_diameter_lst_node.append(np.nanmean(mean_pupil_diameter))

                    source = dfsub['GazeTargetObject'].iloc[i]
                    index -= index  # reset index

            # Last AOI in dataframe (not covert by the loop)
            dfs = dfsub.iloc[i - index: i+1]

            id_lst_node.append(identifier)
            complex_lst_node.append(complexity)
            expert_lst_node.append(expertise)
            interval_lst_node.append(interval)
            duration_start_lst_node.append(dfs['Time'].iloc[0])
            source_lst_node.append(source)

            aoi_duration = dfs['GazeTargetObjectTimes'].iloc[0]
            aoi_duration_lst_node.append(aoi_duration)

            # Clicks on AOI
            if any(dfs['ControllerClicked']):
                clicked_lst_node.append(1)
            else:
                clicked_lst_node.append(0)

            # Pupil diameter
            df_p = pd.DataFrame(
                {'LeftPupilBaselineCorrected': dfs['LeftPupilSize'] - baselines_pupil_diameter[0],
                 'RightPupilBaselineCorrected': dfs['RightPupilSize'] - baselines_pupil_diameter[1]})

            mean_pupil_diameter = np.nanmean(df_p, axis=1)
            pupil_diameter_lst_node.append(np.nanmean(mean_pupil_diameter))


            placeholder_node = [np.nan] * len(id_lst_node)
            placeholder_edge = [np.nan] * len(id_lst_edge)

            #### Create both dataframes (node and egdes) ####
            df_node = pd.DataFrame({'ID': id_lst_node, 'ExpertLevel': expert_lst_node, 'Complexity': complex_lst_node,
                                    '30sInterval': interval_lst_node, 'duration_start': duration_start_lst_node,
                                    'Node': source_lst_node,
                                    'AOI_duration': aoi_duration_lst_node,
                                    'clicked': clicked_lst_node,
                                    'pupil_diameter': pupil_diameter_lst_node,
                                    'controller_duration_on_AOI': controller_duration_on_aoi_lst_node,
                                    'distance_to_aoi': distance_to_aoi_lst_node,
                                    'seating_row_aoi': seating_row_aoi_lst_node,
                                    'seating_loc_aoi': seating_loc_aoi_lst_node,
                                    'controller_direction(pointing)_angle': placeholder_node,
                                    'duration_time_until_first fixation': placeholder_node,
                                    })
            df_trans = pd.DataFrame({'ID': id_lst_edge,  'ExpertLevel': expert_lst_edge, 'Complexity': complex_lst_edge,
                                     '30sTnterval': interval_lst_edge, 'start_transition': trans_start_lst_edge,
                                     'Source': source_lst_edge, 'Target': target_lst_edge, 'Weight': weight_lst_edge,
                                     'trans_duration': trans_dur_lst_edge,
                                     'head_rotation_amplitude': head_amplitude_lst_edge,
                                     'trans_amplitude': placeholder_edge,
                                     'trans_velocity': placeholder_edge,
                                     })

            df_trans.to_csv(save_path + 'ID' + str(identifier) + '_' + str(interval) + '_trans.csv',
                            index=False)
            df_node.to_csv(save_path + 'ID' + str(identifier) + '_' + str(interval) + '_node.csv',
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
