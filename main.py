import os
import pandas as pd
import numpy as np

from utils.S1_eye_tracking_data_aggregation import S101_preprocessing, S102_scanpaths
from utils.S2_graph_dataloader import S201_create_graphs_pytorch
from utils.S2_graph_dataloader import S202_dataloader

from utils.S3_graph_models import GCN_model
from utils.S3_graph_models import GCN_model_node_classification

if __name__ == '__main__':
    project_path = os.path.abspath(os.getcwd())
    # S101_preprocessing.preprocess_data()

    target = 'expertise'  # 'clicked'
    structural_variables = False
    fill_graph_with_zero_nodes = False

    edge_attribute_names = ['trans_duration', 'head_rotation_amplitude', 'trans_amplitude', 'trans_velocity', 'temporal_connect']

    node_attribute_names = ['AOI_duration', 'clicked', 'pupil_diameter', 'controller_duration_on_aoi',
                            'distance_to_aoi', 'seating_row_aoi', 'seating_loc_aoi',
                            'duration_time_until_first_fixation',
                            'active_disruption', 'passive_disruption']

    # S102_scanpaths.create_all_transition_datasets(target)

    S201_create_graphs_pytorch.create_graphs(project_path, target=target, edge_attribute_names=edge_attribute_names,
                                             node_attribute_names=node_attribute_names,
                                             structural_variables=structural_variables,
                                             fill_graph_with_zero_nodes=fill_graph_with_zero_nodes, single_intervals=False)

    dataset = S202_dataloader.load_graphs(project_path, target)

    if target in ['complexity', 'expertise']:
        GCN_model.run_GCN_model(dataset)
    if target in ['disruption', 'clicked']:
        GCN_model_node_classification.run_GCN_model(dataset)
    print(' ')
