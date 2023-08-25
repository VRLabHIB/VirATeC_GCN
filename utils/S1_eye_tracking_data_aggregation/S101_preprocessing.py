import os
import numpy as np
import pandas as pd
from numpy.linalg import norm
from utils import helper


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

        self.df.insert(1, "TimeDiff", helper.calculate_time_diff(self.df))

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
        # The head direction Y angle (angular difference from a straight forward head position)
        # Unity: Left-handed coordinate system with x=left/right, y=up/down, z=forward/backward
        z = np.array([np.zeros(len(self.df)), np.ones(len(self.df))]).T
        v = self.df[['HeadDirectionX', 'HeadDirectionZ']].to_numpy()

        dot = np.einsum('ij,ij->i', z, v)
        no = norm(v, axis=1)
        sign = np.array(np.sign(np.cross(z, v, axis=1))) * (-1)
        angle = np.degrees(np.arccos(dot / no)) * sign

        self.df.insert(8, 'HeadDirectionYAngle', angle)

    def clean_aoi_target(self, target):
        if target == 'gaze':
            object_var = 'GazeTargetObject'
        if target == 'controller':
            object_var = 'ControllerTargetObject'

        lst = list(self.df[self.df[object_var].str.startswith('Child')][object_var].unique())
        lst.append('PresentationBoard')

        self.df[object_var] = np.where(self.df[object_var].isin(lst), self.df[object_var],
                                               'none')
        self.df[object_var] = self.df[object_var].str.replace('Child ', '')
        self.df[object_var] = self.df[object_var].str.replace(' ', '')

    def interpolate_AOI_objects(self, target='gaze'):
        """
        :param target: is either 'gaze' or 'controller'
        :return: self.df processed
        """
        if target == 'gaze':
            object_var = 'GazeTargetObject'
        if target == 'controller':
            object_var = 'ControllerTargetObject'

        object_vector = self.df[object_var].copy()
        # Remove single occurring aois
        if object_vector[0] != object_vector[1]:
            object_vector[0] = 'none'

        if object_vector[len(object_vector) - 1] != object_vector[len(object_vector) - 2]:
            object_vector[len(object_vector) - 1] = 'none'

        for row in range(1, len(self.df) - 1):
            aoi = object_vector[row]
            if aoi != 'none':
                if np.logical_and(aoi != object_vector[row - 1], aoi != object_vector[row + 1]):
                    object_vector[row] = 'none'

        # Interpolate consecutive aois with max break of 2 nones
        # Find first aoi
        row = 0
        while object_vector[row] == 'none':
            row += 1
        # Now interpolate
        while row < len(object_vector):
            if object_vector[row] == 'none':
                last_aoi = object_vector[row - 1]

                length = 0
                while np.logical_and(row < len(object_vector)-1, object_vector[row] == 'none'):
                    row += 1
                    length += 1

                next_aoi = object_vector[row]

                if np.logical_and(last_aoi == next_aoi, length < 3):
                    object_vector[row - length:row] = last_aoi

            row += 1

        self.df[object_var] = object_vector

    def calculate_aoi_times(self, target):
        if target == 'gaze':
            object_var = 'GazeTargetObject'
            column_index = [12, 13]
        if target == 'controller':
            object_var = 'ControllerTargetObject'
            column_index = [19, 20]

        time_lst = list()
        unit_lst = list()
        i = 0
        start_ooi = 'start'
        start_count = 0
        start_time = 0
        while i < len(self.df):
            ooi = self.df[object_var].iloc[i]

            if ooi != start_ooi:
                length = i - start_count
                duration = self.df['Time'].iloc[i] - start_time
                for j in range(length):
                    time_lst.append(duration)
                    unit_lst.append(j + 1)

                start_ooi = self.df[object_var].iloc[i]
                start_count = i
                start_time = self.df['Time'].iloc[i]

            i += 1
        duration = self.df['Time'].iloc[len(self.df) - 1] - start_time
        j = 1
        for k in range(start_count, len(self.df)):
            time_lst.append(duration)
            unit_lst.append(j)
            j += 1

        self.df.insert(column_index[0], object_var + 'Times', time_lst)
        self.df.insert(column_index[1], object_var + 'Unit', unit_lst)

    def calculate_distance_to_gazed_aoi(self):
        start_gaze = self.df[['HeadPositionX', 'HeadPositionY', 'HeadPositionZ']].to_numpy()
        end_gaze = self.df[['GazeHitPointX', 'GazeHitPointY', 'GazeHitPointZ']].to_numpy()
        squared_dist_gaze = np.sum((end_gaze - start_gaze) ** 2, axis=1)
        dist_gaze = np.sqrt(squared_dist_gaze)
        self.df.insert(17, 'RayDistanceGaze', dist_gaze)

        start_controller = self.df[['RightControllerPositionX', 'RightControllerPositionY', 'RightControllerPositionZ']].to_numpy()
        end_controller = self.df[['ControllerHitPointX', 'ControllerHitPointY', 'ControllerHitPointZ']].to_numpy()
        squared_dist_controller = np.sum((end_controller - start_controller) ** 2, axis=1)
        dist_controller = np.sqrt(squared_dist_controller)
        self.df.insert(len(self.df.columns), 'RayDistanceController', dist_controller)

    def add_seating_information(self):
        t = self.df.copy()
        column_index = [14, 15]
        aoi_lst = ['1A', '1B', '2A', '2B', '3A', '3B', '4A', '4B', '5A', '5B', '6A', '6B', '7A', '7B', '8A', '8B',
                    '9A', '9B', 'PresentationBoard']
        row_numbers =[ 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 0]
        loc_numbers = [-1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 1, 1, 0]

        df_seating = pd.DataFrame({'GazeTargetObject': aoi_lst, 'SeatingRowGazeTarget': row_numbers,
                                   'SeatingLocGazeTarget': loc_numbers})

        self.df = self.df.merge(df_seating, on='GazeTargetObject', how='left')

    def add_disengagement_information(self):
        stud_lst = ['1A', '1B', '2A', '2B', '3A', '3B', '4A', '4B', '5A', '5B', '6A', '6B', '7A', '7B', '8A', '8B',
                    '9A', '9B']
        dstr = pd.read_csv(self.project_path + "//metadata//szenarios.csv", sep=";", index_col=False)
        dstr = dstr.T

        for stud in stud_lst:
            self.df[stud] = np.nan

        self.df['SituationalComplexity'] = np.nan

        self.df = mask_disengagement_events(self.df, dstr)

    def add_expertise_levels(self, ID):
        os.chdir(r'V:\VirATeC\data\VirATeC_GCN\2_questionnaire')
        df_q = pd.read_csv(r"pre_post.csv", sep=";", index_col=False, decimal=',', encoding='ISO-8859-1')
        df_q.insert(1, 'Expert?', np.where(df_q['years'] > 0, 1, 0))
        df_q = df_q[['ID', 'Expert?']]

        df_q['ID'] = df_q['ID'].astype(int)

        id_number = int(ID.split('D')[1])

        self.df.insert(len(self.df.columns), 'ExpertLevel',
                       [df_q[df_q['ID'] == id_number]['Expert?'].values[0]] * len(self.df))

    def save(self):
        os.chdir(r'V:\VirATeC\data\VirATeC_GCN\1_full_sessions')
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

        data.clean_aoi_target('gaze')
        data.interpolate_AOI_objects('gaze')
        data.calculate_aoi_times('gaze')
        data.calculate_distance_to_gazed_aoi()

        data.clean_aoi_target('controller')
        data.interpolate_AOI_objects('controller')
        data.calculate_aoi_times('controller')

        data.add_seating_information()
        data.calculate_head_directionY_angle()

        data.add_disengagement_information()
        data.add_expertise_levels(ID)
        df = data.get_data()

        df_stats = helper.calculate_object_stats(df)
        df_stats.to_csv(project_path + '//eval//OOIs//{}_ooi_freq.csv'.format(ID), index=False)

        # Use data when the tracking ratio is over 80%
        if np.logical_and(left_tr >= 0.8, right_tr >= 0.8):
            print('File Saved')
            data.save()

    df_tr = pd.DataFrame({'ID': ID_lst, 'Left TR': left_tr_lst, 'Right TR': right_tr_lst})
    df_tr.to_csv(project_path + '//eval//tracking_ratio//All_tracking_ratios.csv')
