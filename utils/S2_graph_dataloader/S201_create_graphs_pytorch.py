import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data

from utils.helper import locate_transition_data
from utils.helper import locate_node_data
from utils.helper import from_networkx  # undirected transform version modified from:


# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/convert.html

class Datasets:
    def __init__(self, trans_name, node_name, identifier, project_path):
        self.dft = pd.read_csv(trans_name, low_memory=False)
        self.dft = self.dft.rename(columns={'Weight': 'weight'})
        self.dfn = pd.read_csv(node_name, low_memory=False)

        self.ID = identifier
        self.project_path = project_path

        self.G = None
        self.dfg = Data()

    def get_data(self):
        return self.dft

    def get_networkx_graph(self):
        return self.G

    def get_data_graph(self):
        return self.dfg

    def get_ID(self):
        return self.ID

    def get_y(self): #TODO change for comlexity
        #y = int(0)
        #if self.dft['condition'].iloc[0] == 'E':
        #    y = int(0)
        #if self.dft['condition'].iloc[0] == 'C':
        #    y = int(1)
        return int(self.dft['condition'].iloc[0])

    def create_undirected_graph_with_dataframe(self, edge_attr):
        matrix = self.dft[['Source', 'Target']].to_numpy()

        self.dft.insert(2, 'combinations', ['_'.join(np.unique(tupel)) for tupel in matrix])

        self.dft = self.dft.iloc[:, 2:].groupby('combinations').sum().reset_index()
        dfsplit = self.dft['combinations'].str.split('_', expand=True)
        dfsplit.columns = ['Source', 'Target']
        self.dft = self.dft.drop(columns=['combinations'])
        self.dft['Source'], self.dft['Target'] = dfsplit['Source'], dfsplit['Target']

        self.G = nx.from_pandas_edgelist(self.dft, source='Source', target='Target',
                                         edge_attr=edge_attr)

    def create_directed_graph_with_networkx(self, edge_attr):
        self.G = nx.from_pandas_edgelist(self.dft, source='Source', target='Target',
                                         edge_attr=edge_attr, create_using=nx.DiGraph())

    def create_undirected_graph_with_networkx(self, edge_attr):
        D = nx.from_pandas_edgelist(self.dft, source='Source', target='Target',
                                    edge_attr=edge_attr, create_using=nx.DiGraph())
        self.G = D.to_undirected()

        # Transform G into weighted undirected graph
        for node in D:
            for neighbor in nx.neighbors(D, node):
                if node in nx.neighbors(D, neighbor):
                    for attribute in edge_attr:
                        self.G.edges[node, neighbor][attribute] = (D.edges[node, neighbor][attribute]
                                                                   + D.edges[neighbor, node][attribute])

    def add_note_attributes_to_networkx(self, structural_variables):
        if structural_variables is not None:
            # merge node attributes and structura variables
            self.dfn = self.dfn.merge(structural_variables, on='Node', how='left')

        self.dfn.index = self.dfn['Node']
        self.dfn = self.dfn.drop(columns=['Node'])
        self.dfn = self.dfn.transpose()
        dict = self.dfn.to_dict()
        nx.set_node_attributes(self.G, dict)

    def calculate_structural_variables(self, structural_variable_names):
        df_lst = list()
        if 'degree_centrality' in structural_variable_names:
            # Add graph structural variables as node features
            degree_centrality = nx.degree_centrality(self.G)
            degree_centrality = pd.DataFrame.from_dict(degree_centrality, orient='index')
            degree_centrality = degree_centrality.reset_index()
            df_lst.append(degree_centrality)

        if 'node_clique_number' in structural_variable_names:
            node_clique_number = nx.node_clique_number(self.G)
            node_clique_number = pd.DataFrame.from_dict(node_clique_number, orient='index')
            node_clique_number = node_clique_number.reset_index()

        if len(structural_variable_names) == 0:
            return None

        if len(structural_variable_names) == 1:
            df = df_lst[0]
            df.columns = ['Node'] + structural_variable_names
            return df_lst[0]

        if len(structural_variable_names) > 1:
            df = df_lst[0]
            for i in range(1, len(df_lst)):
                df = df.merge(df_lst[i], on='Node')
                df.reset_index()
            df.columns = ['Node'] + structural_variable_names
            return df

    def create_graph(self, edge_attribute_names, node_attribute_names, structural_variable_names):
        y = self.get_y()
        edge_attr = ['weight'] + edge_attribute_names

        aoi_lst = ['1A', '1B', '2A', '2B', '3A', '3B', '4A', '4B', '5A', '5B', '6A', '6B',
                   '7A', '7B', '8A', '8B', '9A', '9B', 'PB']

        # Modify adjacency data frame
        dftt = pd.concat([self.dft[['Source', 'Target']], self.dft[edge_attr]], axis=1)
        self.dft = dftt.groupby(['Source', 'Target']).sum().reset_index()

        # Fill missing edeges with zero
        #for source in aoi_lst:
        #    for target in aoi_lst:
        #        if source != target:
        #            if np.logical_and(source not in self.dft['Source'].values, target not in self.dft['Target'].values):
        #                zeros = [0] * (len(edge_attribute_names) + 1)
        #                self.dft.loc[len(self.dft)] = [source, target] + zeros

        # Modify node dataframe
        dfnn = pd.concat([self.dfn[['Node']], self.dfn[node_attribute_names]], axis=1)
        self.dfn = dfnn.groupby(['Node']).sum().reset_index()

        # Fill missing nodes with zero
        #for aoi in aoi_lst:
        #    if aoi not in self.dfn['Node'].values:
        #        zeros = [0] * len(node_attribute_names)
        #        self.dfn.loc[len(self.dfn)] = [aoi] + zeros

        # Create graph
        self.create_directed_graph_with_networkx(edge_attr)

        # Add graph structural variables as node features
        structural_variables = self.calculate_structural_variables(structural_variable_names)

        # Add further node attributes
        self.add_note_attributes_to_networkx(structural_variables)
        node_attribute_names = node_attribute_names + structural_variable_names

        # Create Data format to save
        data = from_networkx(self.G, group_node_attrs=node_attribute_names, group_edge_attrs=edge_attr)
        data['y'] = y  # Add complexity label
        edge_weights = data['edge_attr'][:, 0]
        data.edge_weight = edge_weights

        return data


def create_graphs(project_path):
    trans_lst = locate_transition_data(project_path)
    node_lst = locate_node_data(project_path)

    print('Create graph Datasets:')

    for i in range(len(trans_lst)):
        trans_name = trans_lst['name'].iloc[i]
        node_name = node_lst['name'].iloc[i]
        identifier = trans_lst['ID'].iloc[i]
        print('ID {}'.format(identifier))

        data = Datasets(trans_name, node_name, identifier, project_path)

        name = trans_name.split('.')[0].rsplit('_', 1)[0]
        if len(data.get_data()) != 0:
            edge_attribute_names = ['trans_duration']
            node_attribute_names = ['AOI_duration', 'clicked', 'pupil_diameter']
            structural_variable_names = ['degree_centrality']  # , 'node_clique_number']
            graph = data.create_graph(edge_attribute_names, node_attribute_names, structural_variable_names)
            torch.save(graph, project_path + '\\data\\graphs\\' + name + '.pt')
