import os
import glob
import pandas as pd
import torch

from utils.helper import locate_graph_data


def load_graphs(project_path):
    data_path = project_path + '\\data\\graphs\\'
    data_lst = locate_graph_data(project_path)

    print('Create Graph Datasets:')
    dataset = list()

    os.chdir(data_path)
    for i in range(len(data_lst)):
        name = data_lst['name'].iloc[i]
        identifier = data_lst['ID'].iloc[i]
        print('ID {}'.format(identifier))

        dataset.append(torch.load(name))

    return dataset
