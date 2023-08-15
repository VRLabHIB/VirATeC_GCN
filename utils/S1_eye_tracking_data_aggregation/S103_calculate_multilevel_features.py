import os
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from utils.S1_eye_tracking_data_aggregation.helper import locate_processed_data


class MultiData:
    def __init__(self, name, ID, project_path):
        os.chdir(r'V:\VirATeC\data\VirATeC_NSR\1_full_sessions')
        self.df = pd.read_csv(name, low_memory=False)
        self.ID = ID
        self.project_path = project_path

        starts = np.arange(0, 601, 30)
        ends = np.arange(30, 631, 30)
        self.students = ['1A', '1B', '2A', '2B', '3A', '3B', '4A', '4B', '5A', '5B', '6A', '6B', '7A', '7B', '8A', '8B',
                         '9A', '9B']

        self.dff = pd.DataFrame({'TID': [], 'SID': [], 'StartInSec': [], 'EndInSec': []})

        for i in range(len(starts)):
            df_s = pd.DataFrame(
                {'TID': [ID] * len(self.students), 'SID': self.students, 'StartInSec': [starts[i]] * len(self.students),
                 'EndInSec': [ends[i]] * len(self.students)})

            self.dff = self.dff.append(df_s)

    def get_data(self):
        return self.dff

    def add_student_seating(self):
        self.dff['SeatingFrontBack'] = np.nan
        self.dff['SeatingLeftRight'] = np.nan

        for stud in self.students:
            mask = (self.dff['SID'] == stud)
            if stud in ['1A', '1B', '2A', '2B', '3A', '3B']:
                self.dff.loc[mask, 'SeatingFrontBack'] = 'front'
            if stud in ['4A', '4B', '5A', '5B', '6A', '6B']:
                self.dff.loc[mask, 'SeatingFrontBack'] = 'mid'
            if stud in ['7A', '7B', '8A', '8B', '9A', '9B']:
                self.dff.loc[mask, 'SeatingFrontBack'] = 'back'

            if stud in ['1A', '1B', '4A', '4B', '7A', '7B']:
                self.dff.loc[mask, 'SeatingLeftRight'] = 'left'
            if stud in ['2A', '2B', '5A', '5B', '8A', '8B']:
                self.dff.loc[mask, 'SeatingLeftRight'] = 'mid'
            if stud in ['3A', '3B', '6A', '6B', '9A', '9B']:
                self.dff.loc[mask, 'SeatingLeftRight'] = 'right'

    def calculate_disturbances(self):
        self.dff['ActiveDisruptions'] = np.nan
        self.dff['PassiveDisruptions'] = np.nan

        for s, e in zip(self.dff['StartInSec'], self.dff['EndInSec']):
            df_sub = self.df[np.logical_and(self.df['Time'] >= s, self.df['Time'] < e)]
            active_lst, passive_lst = get_active_passives(df_sub)

            mask = (self.dff['StartInSec'] == s) & (self.dff['SID'].isin(active_lst))
            self.dff.loc[mask, 'ActiveDisruptions'] = 1

            mask = (self.dff['StartInSec'] == s) & (self.dff['SID'].isin(passive_lst))
            self.dff.loc[mask, 'PassiveDisruptions'] = 1

    def calculate_stopping(self):
        self.dff['SuccessfullyActiveStopped'] = np.nan
        self.dff['SuccessfullyPassiveStopped'] = np.nan
        self.dff['FalselyClicked'] = np.nan

        self.df['ControllerClicked'] = self.df['ControllerClicked'].astype(bool)

        for s, e in zip(self.dff['StartInSec'], self.dff['EndInSec']):
            df_sub = self.df[np.logical_and(self.df['Time'] >= s, self.df['Time'] < e)]

            active_lst, passive_lst = get_active_passives(df_sub)
            last_click = 0
            last_stud = 'none'

            df_c = df_sub[df_sub['ControllerClicked']]

            for i in range(len(df_c)):
                object = df_c['ControllerTargetObject'].iloc[i]

                if np.logical_and(object != 'PresentationBoard', object != 'none'):
                    if np.logical_and(object in active_lst,
                                      np.logical_or(last_stud != object, df_c['Time'].iloc[i] >= last_click + 15)):
                        mask = (self.dff['StartInSec'] == s) & (self.dff['SID'] == object)
                        self.dff.loc[mask, 'SuccessfullyActiveStopped'] = 1

                        last_click = df_c['Time'].iloc[i]
                        last_stud = object

                    if np.logical_and(object in passive_lst,
                                      np.logical_or(last_stud != object, df_c['Time'].iloc[i] >= last_click + 15)):
                        mask = (self.dff['StartInSec'] == s) & (self.dff['SID'] == object)
                        self.dff.loc[mask, 'SuccessfullyPassiveStopped'] = 1

                        last_click = df_c['Time'].iloc[i]
                        last_stud = object

                    if np.logical_and(np.logical_and(object not in active_lst, object not in passive_lst),
                                      np.logical_or(last_stud != object, df_c['Time'].iloc[i] >= last_click + 15)):
                        mask = (self.dff['StartInSec'] == s) & (self.dff['SID'] == object)
                        self.dff.loc[mask, 'FalselyClicked'] = 1

                        last_click = df_c['Time'].iloc[i]
                        last_stud = object


def create_multilevel_student_dataset(project_path):
    df = pd.read_csv(project_path + '//data//FeatureDataset.csv')
    data_lst = locate_processed_data()

    df_multi_features = pd.DataFrame()
    for i in range(len(data_lst)):
        print('Name: ', name)
        print('ID:   ', ID)
        name = data_lst['name'].iloc[i]
        ID = data_lst['ID'].iloc[i]

        data = MultiData(name, ID, project_path)
        data.add_student_seating()
        data.calculate_disturbances()
        data.calculate_stopping()
        dff = data.get_data()

        df_multi_features = df_multi_features.append(dff)
    df_multi_features.to_csv(project_path + '//data//FeatureDatasetMultilevel.csv', index=False)


def merge_multilevel_datasets(project_path):
    df_s = pd.read_csv(project_path + '//data//FeatureDatasetMultilevel.csv')
    df_e = pd.read_csv(project_path + '//data//FeatureDataset.csv')
    df_e = df_e.rename(columns={'ID': 'TID'})

    df_f = df_s.merge(df_e, on=['TID', 'StartInSec', 'EndInSec'])
    df_f.to_csv(project_path + '//data//FeatureDatasetFullMultilevel.csv', index=False)



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
