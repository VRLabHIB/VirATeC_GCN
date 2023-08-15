import os
import glob
import pandas as pd
import numpy as np

from utils.S1_eye_tracking_data_aggregation import helper
from utils.S1_eye_tracking_data_aggregation import S101_preprocessing
from utils.S1_eye_tracking_data_aggregation import S102_calculate_features
from utils.S2_questionnaire_items import S201_expertise_levels
from utils.S1_eye_tracking_data_aggregation import S103_calculate_multilevel_features
from utils.S1_eye_tracking_data_aggregation import S104_scanpaths
from utils.S1_eye_tracking_data_aggregation import S105_adjacency_matix
from utils.S1_eye_tracking_data_aggregation import S106_graph_features


def add_expertise_levels(dff):
    import os
    os.chdir(r'V:\VirATeC\data\VirATeC_NSR\2_questionnaire')
    df_q = pd.read_csv(r"pre_post.csv", sep=";", index_col=False, decimal=',', encoding='ISO-8859-1')
    df_q.insert(1, 'Expert?', np.where(df_q['years'] > 0, 1, 0))
    df_q = df_q[['ID', 'Expert?']]

    ID_lst = list()
    for i in range(len(dff)):
        ID = dff['ID'].iloc[i]
        ID = ID[2] + ID[3] + ID[4]
        ID = int(ID)
        ID_lst.append(ID)
    dff = dff.drop(columns=['ID'])
    dff.insert(0, 'ID', ID_lst)

    dff = dff.merge(df_q, on='ID')
    expert = dff['Expert?'].values
    dff = dff.drop(columns=['Expert?'])
    dff.insert(4, 'Expert?', expert)

    return dff

if __name__ == '__main__':
    project_path = os.path.abspath(os.getcwd())
    S101_preprocessing.preprocess_data()
    S102_calculate_features.create_long_formats()
    print(' ')
    # S201_expertise_levels.add_expertise_levels(project_path=project_path, multilevel=False)

    # S103_calculate_multilevel_features.create_multilevel_student_dataset(project_path)
    # S103_calculate_multilevel_features.merge_multilevel_datasets(project_path)
    # S201_expertise_levels.add_expertise_levels_multilevel(project_path=project_path, multilevel=True)

    #S104_scanpaths.create_scanpaths()
    #mat_lst, graph_list, cond_lst, ID_lst = S105_adjacency_matix.create_matrices_and_graphs()

    #S106_graph_features.create_feature_dataframe

    data_path = project_path + '//data//'
    #df_input = pd.read_csv(data_path + 'transitions.csv')
    df_input = pd.read_csv(data_path + 'FeatureDataset.csv')

    df_input = add_expertise_levels(df_input)

    print(' ')
