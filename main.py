import os
import pandas as pd
import numpy as np


from utils.S1_eye_tracking_data_aggregation import S101_preprocessing, S102_scanpaths
from utils.S2_graph_dataloader import S201_create_graphs_pytorch
from utils.S2_graph_dataloader import S202_dataloader

from utils.S3_graph_models import GCN_model

if __name__ == '__main__':
    project_path = os.path.abspath(os.getcwd())
    S101_preprocessing.preprocess_data()

    S102_scanpaths.create_all_transition_datasets()

    # S201_create_graphs_pytorch.create_graphs(project_path)

    dataset = S202_dataloader.load_graphs(project_path)

    GCN_model.run_GCN_model(dataset)
    print(' ')



    # mat_lst, graph_list, cond_lst, ID_lst = S105_adjacency_matix.create_matrices_and_graphs()

    # S106_graph_features.create_feature_dataframe

    #data_path = project_path + '//data//'
    #df_input = pd.read_csv(data_path + 'nodes_and_transitions.csv')
    # df_input = pd.read_csv(data_path + 'FeatureDataset.csv')

    #df_input = add_expertise_levels(df_input)

    print(' ')
