import os

import pandas as pd
import numpy as np

from utils.helper import locate_processed_data


# Class to calculate features from the dataframes / processing pipline is stated below
class FeatureDataset:
    def __init__(self, name, identifier, project_path):
        os.chdir(r'V:\VirATeC\data\VirATeC_NSR\1_full_sessions')
        self.df = pd.read_csv(name, low_memory=False)
        self.ID = identifier
        self.project_path = project_path

        starts = np.arange(0, 601, 30)
        ends = np.arange(30, 631, 30)
        self.dff = pd.DataFrame({'ID': [identifier] * len(starts), 'StartInSec': starts, 'EndInSec': ends})

    def get_full(self):
        return self.df

    def get_long(self):
        return self.dff

    def get_ID(self):
        return self.ID

    def add_session_complexity(self):
        sit_com_lst = list()
        for s, e in zip(self.dff['StartInSec'], self.dff['EndInSec']):
            compl = self.df[np.logical_and(self.df['Time'] >= s, self.df['Time'] < e)]['SituationalComplexity'].iloc[0]
            sit_com_lst.append(compl)
        self.dff['SituationalComplexity'] = sit_com_lst

    def add_disruptor_numbers(self):
        np_lst = list()
        na_lst = list()
        for s, e in zip(self.dff['StartInSec'], self.dff['EndInSec']):
            df_sub = self.df[np.logical_and(self.df['Time'] >= s, self.df['Time'] < e)]

            series = df_sub.iloc[0, list(self.df.columns).index('1A'):list(self.df.columns).index('9B') + 1].values
            x = list(filter(lambda v: v == v, series))
            np_lst.append(sum('P' in mode for mode in x))   # Passive
            na_lst.append(sum('IA' in mode for mode in x))  # Active

        self.dff['TotalActiveDisruptors'] = na_lst
        self.dff['TotalPassiveDisruptors'] = np_lst

    def calculate_head_movements(self):
        max_left_right_lst = list()
        mean_left_right_lst = list()
        max_back_forth_lst = list()
        mean_back_forth_lst = list()

        for s, e in zip(self.dff['StartInSec'], self.dff['EndInSec']):
            df_sub = self.df[np.logical_and(self.df['Time'] >= s, self.df['Time'] < e)]

            max_left_right_lst.append(np.abs(np.max(df_sub['HeadPositionX']) - np.min(df_sub['HeadPositionX'])))
            t1 = df_sub['HeadPositionX'].iloc[1:].values
            t0 = df_sub['HeadPositionX'].iloc[0:-1].values

            dist = np.divide(np.abs(t1 - t0), df_sub['TimeDiff'].iloc[1:])
            mean_left_right_lst.append(np.nanmean(dist))

            max_back_forth_lst.append(np.abs(np.max(df_sub['HeadPositionZ']) - np.min(df_sub['HeadPositionZ'])))
            t1 = df_sub['HeadPositionZ'].iloc[1:].values
            t0 = df_sub['HeadPositionZ'].iloc[0:-1].values
            dist = np.divide(np.abs(t1 - t0), df_sub['TimeDiff'].iloc[1:])
            mean_back_forth_lst.append(np.nanmean(dist))

        self.dff['MaxDistanceLeftRightMovement'] = max_left_right_lst
        self.dff['MeanLeftRightMovement'] = mean_left_right_lst
        self.dff['MaxDistanceBackForthMovement'] = max_back_forth_lst
        self.dff['MeanBackForthMovement'] = mean_back_forth_lst

    def calculate_head_rotation(self):
        head_rot_ampl_lst = list()
        head_rot_mean_lst = list()

        for s, e in zip(self.dff['StartInSec'], self.dff['EndInSec']):
            df_sub = self.df[np.logical_and(self.df['Time'] >= s, self.df['Time'] < e)]
            ampl = np.abs(np.max(df_sub['HeadDirectionYAngle']) - np.min(df_sub['HeadDirectionYAngle']))
            head_rot_ampl_lst.append(ampl)

            t1 = df_sub['HeadDirectionYAngle'].iloc[1:].values
            t0 = df_sub['HeadDirectionYAngle'].iloc[0:-1].values

            dist = np.divide(np.abs(t1 - t0), df_sub['TimeDiff'].iloc[1:])
            head_rot_mean_lst.append(np.nanmean(dist))

        self.dff['AmplitudeLeftRightHeadRotation'] = head_rot_ampl_lst
        self.dff['MeanLeftRightHeadRotation'] = head_rot_mean_lst

    def gaze_object_durations(self):
        gazed_board = list()
        gazed_student = list()
        gazed_active = list()
        gazed_passive = list()

        for s, e in zip(self.dff['StartInSec'], self.dff['EndInSec']):
            df_sub = self.df[np.logical_and(self.df['Time'] >= s, self.df['Time'] < e)]
            df_sub = df_sub[df_sub['GazeTargetObject'] != 'none']

            t_board = \
                df_sub[
                    np.logical_and(df_sub['GazeTargetObject'] == 'PresentationBoard', df_sub['GazeTargetUnit'] == 1)][
                    'GazeTargetTimes'].sum()
            gazed_board.append(t_board)

            df_stud = df_sub[df_sub['GazeTargetObject'] != 'PresentationBoard']
            gazed_student.append(df_stud[df_stud['GazeTargetUnit'] == 1]['GazeTargetTimes'].sum())

            active_lst, passive_lst = get_active_passives(df_sub)
            df_active = df_stud[df_stud['GazeTargetObject'].isin(active_lst)]
            df_passive = df_stud[df_stud['GazeTargetObject'].isin(passive_lst)]
            if len(df_active) > 0:
                gazed_active.append(df_active[df_active['GazeTargetUnit'] == 1]['GazeTargetTimes'].sum())
            if len(df_active) == 0:
                gazed_active.append(0)

            if len(df_passive) > 0:
                gazed_passive.append(df_passive[df_passive['GazeTargetUnit'] == 1]['GazeTargetTimes'].sum())
            if len(df_passive) == 0:
                gazed_passive.append(0)

        self.dff['GazeDurationOnBoard'] = gazed_board
        self.dff['GazeDurationOnStudents'] = gazed_student
        self.dff['GazeDurationOnActiveStudents'] = gazed_active
        self.dff['GazeDurationOnPassiveStudents'] = gazed_passive

    def calculate_clicking(self):
        n_total_clicks_lst = list()
        n_false_clicks_on_stud_lst = list()
        n_successful_active_clicks_lst = list()
        n_successful_passive_clicks_lst = list()

        self.df['ControllerClicked'] = self.df['ControllerClicked'].astype(bool)
        for s, e in zip(self.dff['StartInSec'], self.dff['EndInSec']):
            df_sub = self.df[np.logical_and(self.df['Time'] >= s, self.df['Time'] < e)]
            active_lst, passive_lst = get_active_passives(df_sub)
            n_total_clicks = 0
            n_false_clicks_on_stud = 0
            n_successful_active_clicks = 0
            n_successful_passive_clicks = 0
            last_click = 0
            last_stud = 'none'

            df_c = df_sub[df_sub['ControllerClicked']]

            for i in range(len(df_c)):
                cobject = df_c['ControllerTargetObject'].iloc[i]
                if cobject == 'none':
                    n_total_clicks += 1
                if cobject == 'PresentationBoard':
                    n_total_clicks += 1
                if np.logical_and(cobject != 'PresentationBoard', cobject != 'none'):
                    if np.logical_and(cobject in active_lst,
                                      np.logical_or(last_stud != cobject, df_c['Time'].iloc[i] >= last_click + 15)):
                        n_total_clicks += 1
                        n_successful_active_clicks += 1
                        last_click = df_c['Time'].iloc[i]
                        last_stud = cobject
                    if np.logical_and(cobject in passive_lst,
                                      np.logical_or(last_stud != cobject, df_c['Time'].iloc[i] >= last_click + 15)):
                        n_total_clicks += 1
                        n_successful_passive_clicks += 1
                        last_click = df_c['Time'].iloc[i]
                        last_stud = cobject
                    if np.logical_and(np.logical_and(cobject not in active_lst, cobject not in passive_lst),
                                      np.logical_or(last_stud != cobject, df_c['Time'].iloc[i] >= last_click + 15)):
                        n_total_clicks += 1
                        n_false_clicks_on_stud += 1
                        last_click = df_c['Time'].iloc[i]
                        last_stud = cobject

            n_total_clicks_lst.append(n_total_clicks)
            n_false_clicks_on_stud_lst.append(n_false_clicks_on_stud)
            n_successful_active_clicks_lst.append(n_successful_active_clicks)
            n_successful_passive_clicks_lst.append(n_successful_passive_clicks)

        self.dff['NumberOfTotalClicks'] = n_total_clicks_lst
        self.dff['NumberOfFalseStudentClicks'] = n_false_clicks_on_stud_lst
        self.dff['NumberOfSuccessfulActiveClicks'] = n_successful_active_clicks_lst
        self.dff['NumberOfSuccessfulPassiveClicks'] = n_successful_passive_clicks_lst


def get_active_passives(df):
    events = df.iloc[0:1, list(df.columns).index('1A'):list(df.columns).index('9B') + 1]
    passive_lst = list()
    active_lst = list()

    for name in events.columns:
        if str(events[name].iloc[0]).startswith('P'):
            passive_lst.append(name)
        if str(events[name].iloc[0]).startswith('IA'):
            active_lst.append(name)

    return active_lst, passive_lst


def calculate_elliptical_movement_area(df):
    df.insert(10, 'EllipticalMovementArea', (df['MaxDistanceLeftRightMovement'] / 2) * (
            df['MaxDistanceBackForthMovement'] / 2) * np.pi)

    return df


########################################################################################################################
# ####################################### Processing pipline ###########################################################
########################################################################################################################
def create_long_formats():
    project_path = os.path.abspath(os.getcwd())
    data_lst = locate_processed_data()

    df_features = pd.DataFrame()
    for i in range(len(data_lst)):
        name = data_lst['name'].iloc[i]
        identifier = data_lst['ID'].iloc[i]

        data = FeatureDataset(name, identifier, project_path)

        data.add_session_complexity()
        data.add_disruptor_numbers()
        data.calculate_head_movements()
        data.calculate_head_rotation()
        data.gaze_object_durations()
        data.calculate_clicking()

        df = data.get_full()
        dff = data.get_long()

        dff = calculate_elliptical_movement_area(dff)

        df_features = pd.concat([df_features, dff], axis=0)

    df_features.to_csv(project_path + '//data//FeatureDataset.csv', index=False)
