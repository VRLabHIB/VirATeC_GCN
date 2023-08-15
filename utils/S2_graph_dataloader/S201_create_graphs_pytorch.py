import os
import pandas as pd
import networkx as nx
import torch

from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx

from utils.helper import locate_transition_data


class TransitionDataset:
    def __init__(self, name, identifier, project_path):
        self.df = pd.read_csv(name, low_memory=False)
        self.ID = identifier
        self.project_path = project_path
        self.df_adj = pd.DataFrame()
        self.dfg = Data()

    def get_data(self):
        return self.df

    def get_graph(self):
        return self.dfg

    def get_ID(self):
        return self.ID

    def get_y(self):
        y = int(0)
        if self.df['condition'].iloc[0] == 'E':
            y = int(0)
        if self.df['condition'].iloc[0] == 'C':
            y = int(1)
        return y

    def create_directed_graph_with_networkx(self, edge_attr):
        D = nx.from_pandas_edgelist(self.df_adj, source='Source', target='Target',
                                    edge_attr=edge_attr, create_using=nx.DiGraph())
        return D

    def create_undirected_graph_with_networkx(self, edge_attr):
        D = nx.from_pandas_edgelist(self.df_adj, source='Source', target='Target',
                                    edge_attr=edge_attr, create_using=nx.DiGraph())
        G = D.to_undirected()

        # Transform G into weighted undirected graph
        for node in D:
            for neighbor in nx.neighbors(D, node):
                if node in nx.neighbors(D, neighbor):
                    for attribute in edge_attr:
                        G.edges[node, neighbor][attribute] = (D.edges[node, neighbor][attribute]
                                                              + D.edges[neighbor, node][attribute])
        return G

    def create_graph(self, attribute_list):
        y = self.get_y()
        edge_attr = ['Weight'] + attribute_list

        # Create adjacency data frame
        dfs = pd.concat([self.df.iloc[:, 4:7], self.df[attribute_list]], axis=1)
        self.df_adj = dfs.groupby(['Source', 'Target']).sum().reset_index()

        # Create graph
        # G = self.create_undirected_graph_with_networkx(edge_attr)
        D = self.create_directed_graph_with_networkx(edge_attr)
        data = from_networkx(D, group_edge_attrs=edge_attr)
        data['y'] = y   # Add complexity label
        return data


def create_graphs(project_path):
    data_lst = locate_transition_data(project_path)

    print('Create graph Datasets:')

    for i in range(len(data_lst)):
        name = data_lst['name'].iloc[i]
        identifier = data_lst['ID'].iloc[i]
        print('ID {}'.format(identifier))

        data = TransitionDataset(name, identifier, project_path)

        name = name.split('.')[0]
        if len(data.get_data()) != 0:
            attribute_list = ['trans_duration']
            graph = data.create_graph(attribute_list)
            torch.save(graph, project_path + '\\data\\graphs\\' + name)
