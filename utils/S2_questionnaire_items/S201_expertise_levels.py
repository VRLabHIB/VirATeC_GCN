import os
import numpy as np
import pandas as pd


def add_expertise_levels():
    project_path = os.path.abspath(os.getcwd())
    dff = pd.read_csv(project_path + '//data//FeatureDataset.csv')

    os.chdir(r'V:\VirATeC\data\VirATeC_NSR\2_questionnaire')
    df_q = pd.read_csv(r"pre_post.csv", sep=";", index_col=False, decimal=',', encoding='ISO-8859-1')
    df_q.insert(1, 'Expert?', np.where(df_q['years'] > 0, 1, 0))
    df_q = df_q[['ID', 'Expert?']]

    df_q['ID'] = df_q['ID'].astype(str)

    for i in range(len(df_q)):
        if len(df_q['ID'].iloc[i]) == 1:
            df_q['ID'].iloc[i] = 'ID00' + df_q['ID'].iloc[i]
        if len(df_q['ID'].iloc[i]) == 2:
            df_q['ID'].iloc[i] = 'ID0' + df_q['ID'].iloc[i]

    dff = dff.merge(df_q, on='ID')

    dff.to_csv(project_path + '//data//FeatureDatasetWithExpertLabels.csv', index=False)


def add_expertise_levels_multilevel(project_path, multilevel):
    if not multilevel:
        dff = pd.read_csv(project_path + '//data//FeatureDatasetMultilevel.csv')

        os.chdir(r'V:\VirATeC\data\VirATeC_NSR\2_questionnaire')
        df_q = pd.read_csv(r"pre_post.csv", sep=";", index_col=False, decimal=',', encoding='ISO-8859-1')
        df_q.insert(1, 'Expert?', np.where(df_q['years'] > 0, 1, 0))
        df_q = df_q[['ID', 'Expert?']]

        df_q['ID'] = df_q['ID'].astype(str)

        for i in range(len(df_q)):
            if len(df_q['ID'].iloc[i]) == 1:
                df_q['ID'].iloc[i] = 'ID00' + df_q['ID'].iloc[i]
            if len(df_q['ID'].iloc[i]) == 2:
                df_q['ID'].iloc[i] = 'ID0' + df_q['ID'].iloc[i]

        dff = dff.merge(df_q, on='ID')

        dff.to_csv(project_path + '//data//FeatureDatasetMultilevelWithExpertLabels.csv', index=False)
    if multilevel:
        dff = pd.read_csv(project_path + '//data//FeatureDatasetFullMultilevel.csv')

        os.chdir(r'V:\VirATeC\data\VirATeC_NSR\2_questionnaire')
        df_q = pd.read_csv(r"pre_post.csv", sep=";", index_col=False, decimal=',', encoding='ISO-8859-1')
        df_q.insert(1, 'Expert?', np.where(df_q['years'] > 0, 1, 0))
        df_q = df_q[['ID', 'Expert?']]

        df_q['ID'] = df_q['ID'].astype(str)

        for i in range(len(df_q)):
            if len(df_q['ID'].iloc[i]) == 1:
                df_q['ID'].iloc[i] = 'ID00' + df_q['ID'].iloc[i]
            if len(df_q['ID'].iloc[i]) == 2:
                df_q['ID'].iloc[i] = 'ID0' + df_q['ID'].iloc[i]

        df_q = df_q.rename(columns={'ID': 'TID'})
        dff = dff.merge(df_q, on='TID')

        dff.to_csv(project_path + '//data//FeatureDatasetFullMultilevelWithExpertLabels.csv', index=False)
