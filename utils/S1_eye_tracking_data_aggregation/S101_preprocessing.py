import os
import numpy as np
import pandas as pd
from numpy.linalg import norm
from src.S1_eye_tracking_data_aggregation import helper as hf, helper


# Class to process the dataframes / processing pipline is stated below
class Data:
    def __init__(self, name, ID, project_path):
        os.chdir(r'V:\VirATeC\data\VirATeC_NSR\0_raw_data')
        self.df = pd.read_csv(name, sep=';', decimal=",", header=0, index_col=False)
        self.name = name
        self.ID = ID
        self.project_path = project_path

    def get_data(self):
        return self.df

    def get_name(self):
        return self.name

    def drop_unused_variable(self):
        self.df = self.df.drop(columns=['SystemTime', 'CurrentSlideIndex', 'GazeOriginX', 'GazeOriginY', 'GazeOriginZ',
                                        'LeftGazeOriginX', 'LeftGazeOriginY', 'LeftGazeOriginZ',
                                        'RightGazeOriginX', 'RightGazeOriginY', 'RightGazeOriginZ',
                                        'LeftGazeDirectionX', 'LeftGazeDirectionY', 'LeftGazeDirectionZ',
                                        'RightGazeDirectionX', 'RightGazeDirectionY', 'RightGazeDirectionZ',
                                        'LeftPupilPositionX', 'LeftPupilPositionY', 'RightPupilPositionX',
                                        'RightPupilPositionY', 'Active_Scene'])

        self.df['ControllerHitPointX'] = self.df['ControllerHitPointX'].astype(float).replace(-99, np.nan)
        self.df['ControllerHitPointY'] = self.df['ControllerHitPointY'].astype(float).replace(-99, np.nan)
        self.df['ControllerHitPointZ'] = self.df['ControllerHitPointZ'].astype(float).replace(-99, np.nan)

    def refactor_time(self):
        self.df = self.df.rename(columns={'TimeStamp': 'Time'})
        self.df['Time'] = self.df['Time'].astype(float)
        self.df = self.df[self.df['Time'] <= 630]

        self.df.insert(1, "TimeDiff", hf.calculate_time_diff(self.df))

    def filter_blinks(self):
        self.df['LeftPupilSize'] = np.where(self.df['LeftPupilSize'] <= 0, np.nan, self.df['LeftPupilSize'])
        self.df['RightPupilSize'] = np.where(self.df['RightPupilSize'] <= 0, np.nan, self.df['RightPupilSize'])

        # Eye openness threshold set to 75% open eye
        self.df['LeftPupilSize'] = np.where(self.df['LeftEyeOpenness'] < 0.75, np.nan, self.df['LeftPupilSize'])
        self.df['RightPupilSize'] = np.where(self.df['LeftEyeOpenness'] < 0.75, np.nan, self.df['RightPupilSize'])

        cond = np.logical_or(self.df['LeftPupilSize'].isna(), self.df['RightPupilSize'].isna())

        self.df['GazeDirectionX'] = np.where(cond, np.nan, self.df['GazeDirectionX'])
        self.df['GazeDirectionY'] = np.where(cond, np.nan, self.df['GazeDirectionY'])
        self.df['GazeDirectionZ'] = np.where(cond, np.nan, self.df['GazeDirectionZ'])
        self.df['GazeTargetObject'] = np.where(cond, 'none', self.df['GazeTargetObject'])
        self.df['GazeHitPointX'] = np.where(cond, np.nan, self.df['GazeHitPointX'])
        self.df['GazeHitPointY'] = np.where(cond, np.nan, self.df['GazeHitPointY'])
        self.df['GazeHitPointZ'] = np.where(cond, np.nan, self.df['GazeHitPointZ'])

        left_tr = np.round(1 - (self.df['LeftPupilSize'].isna().sum() / len(self.df)), 3)
        right_tr = np.round(1 - (self.df['RightPupilSize'].isna().sum() / len(self.df)), 3)
        print('TR Left  Eye ', str(left_tr))
        print('TR Right Eye ', str(right_tr))

        return left_tr, right_tr

    def calculate_head_directionY_angle(self):
        z = np.array([np.zeros(len(self.df)), np.ones(len(self.df))]).T
        v = self.df[['HeadDirectionX', 'HeadDirectionZ']].to_numpy()

        dot = np.einsum('ij,ij->i', z, v)
        no = norm(v, axis=1)
        sign = np.array(np.sign(np.cross(z, v, axis=1))) * (-1)
        angle = np.degrees(np.arccos(dot / no)) * sign

        self.df.insert(8, 'HeadDirectionYAngle', angle)

    def clean_gaze_target(self):
        lst = list(self.df[self.df['GazeTargetObject'].str.startswith('Child')]['GazeTargetObject'].unique())
        lst.append('PresentationBoard')

        self.df['GazeTargetObject'] = np.where(self.df['GazeTargetObject'].isin(lst), self.df['GazeTargetObject'],
                                               'none')
        self.df['GazeTargetObject'] = self.df['GazeTargetObject'].str.replace('Child ', '')
        self.df['GazeTargetObject'] = self.df['GazeTargetObject'].str.replace(' ', '')

    def clean_controller_target(self):
        lst = list(
            self.df[self.df['ControllerTargetObject'].str.startswith('Child')]['ControllerTargetObject'].unique())
        lst.append('PresentationBoard')

        self.df['ControllerTargetObject'] = np.where(self.df['ControllerTargetObject'].isin(lst),
                                                     self.df['ControllerTargetObject'],
                                                     'none')
        self.df['ControllerTargetObject'] = self.df['ControllerTargetObject'].str.replace('Child ', '')
        self.df['ControllerTargetObject'] = self.df['ControllerTargetObject'].str.replace(' ', '')

    def calculate_gaze_times(self):
        gaze_time_lst = list()
        gaze_unit_lst = list()
        i = 0
        start_ooi = 'start'
        start_count = 0
        start_time = 0
        while i < len(self.df):
            ooi = self.df['GazeTargetObject'].iloc[i]

            if ooi != start_ooi:
                length = i - start_count
                duration = self.df['Time'].iloc[i] - start_time
                for j in range(length):
                    gaze_time_lst.append(duration)
                    gaze_unit_lst.append(j + 1)

                start_ooi = self.df['GazeTargetObject'].iloc[i]
                start_count = i
                start_time = self.df['Time'].iloc[i]

            i += 1
        duration = self.df['Time'].iloc[len(self.df) - 1] - start_time
        j = 1
        for k in range(start_count, len(self.df)):
            gaze_time_lst.append(duration)
            gaze_unit_lst.append(j)
            j += 1

        self.df.insert(12, 'GazeTargetTimes', gaze_time_lst)
        self.df.insert(13, 'GazeTargetUnit', gaze_unit_lst)

    def add_disengagement_information(self):
        stud_lst = ['1A', '1B', '2A', '2B', '3A', '3B', '4A', '4B', '5A', '5B', '6A', '6B', '7A', '7B', '8A', '8B',
                    '9A', '9B']
        dstr = pd.read_csv(self.project_path + "//metadata//szenarios.csv", sep=";", index_col=False)
        dstr = dstr.T

        for stud in stud_lst:
            self.df[stud] = np.nan

        self.df['SituationalComplexity'] = np.nan

        self.df = mask_disengagement_events(self.df, dstr)

    def save(self):
        os.chdir(r'V:\VirATeC\data\VirATeC_NSR\1_full_sessions')
        self.df.to_csv(self.ID + '.csv', index=False)


def mask_disengagement_events(df, dstr):
    for i, start, end in zip(np.arange(20), np.arange(0, 571, 30), np.arange(30, 601, 30)):
        mask = (df['Time'] >= start) & (df['Time'] < end)
        df.loc[mask, slice('1A', '9B', 1)] = dstr.iloc[i].values

    mask = df['Time'] < 120
    df.loc[mask, 'SituationalComplexity'] = "easy"
    mask = (df['Time'] >= 120) & (df['Time'] < 210)
    df.loc[mask, 'SituationalComplexity'] = "complex"
    mask = (df['Time'] >= 210) & (df['Time'] < 300)
    df.loc[mask, 'SituationalComplexity'] = "easy"
    mask = (df['Time'] >= 300) & (df['Time'] < 510)
    df.loc[mask, 'SituationalComplexity'] = "complex"
    mask = df['Time'] >= 510
    df.loc[mask, 'SituationalComplexity'] = "easy"

    return df


# Processing pipline
#######################################################################
def preprocess_data():
    project_path = os.path.abspath(os.getcwd())
    data_lst = helper.locate_raw_data()

    index_start = 0
    left_tr_lst = list()
    right_tr_lst = list()
    ID_lst = list()

    for index in range(index_start, len(data_lst)):
        name = data_lst['name'].iloc[index]
        ID = data_lst['ID'].iloc[index]
        ID_lst.append(ID)

        print('Index    ' + str(index))
        print('ID       ' + str(ID))
        print('FileName ' + str(name))

        data = Data(name, ID, project_path)
        data.drop_unused_variable()
        data.refactor_time()

        left_tr, right_tr = data.filter_blinks()
        left_tr_lst.append(left_tr)
        right_tr_lst.append(right_tr)

        data.clean_gaze_target()
        data.calculate_gaze_times()
        data.clean_controller_target()
        data.calculate_head_directionY_angle()

        data.add_disengagement_information()
        df = data.get_data()

        df_stats = helper.calculate_object_stats(df)
        df_stats.to_csv(project_path + '//eval//OOIs//{}_ooi_freq.csv'.format(ID), index=False)

        # Use data when the tracking ratio is over 80%
        if np.logical_and(left_tr >= 0.8, right_tr >= 0.8):
            print('File Saved')
            data.save()

    df_tr = pd.DataFrame({'ID': ID_lst, 'Left TR': left_tr_lst, 'Right TR': right_tr_lst})
    df_tr.to_csv(project_path + '//eval//tracking_ratio//All_tracking_ratios.csv')
