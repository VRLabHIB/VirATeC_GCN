import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data

from utils.helper import locate_transition_data, delete_files_in_directory
from utils.helper import locate_node_data
from utils.helper import from_networkx  # undirected transform version modified from:


# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/convert.html

def load_transition_dataset(trans_names, identifier):
    aoi_lst = ['1A', '1B', '2A', '2B', '3A', '3B', '4A', '4B', '5A', '5B', '6A', '6B',
               '7A', '7B', '8A', '8B', '9A', '9B', 'PB']

    trans_names = trans_names[trans_names['IDint'] == identifier]

    intervals = np.sort(trans_names['Interval'].unique().astype(int)).astype(str)
    for i in range(len(trans_names)):
        filename = trans_names[trans_names['Interval'] == intervals[i]]['name'].values
        if i == 0:
            dft = pd.read_csv(filename[0], low_memory=False)
            dft = dft.rename(columns={'Weight': 'weight'})
            dft = dft.drop(columns=['Complexity', '30sTnterval', 'start_transition'])
            interval1 = intervals[i]
            dft['Source'] = dft['Source'].values + str(intervals[i])
            dft['Target'] = dft['Target'].values + str(intervals[i])
            dft['temporal_connect'] = 0

        if i != 0:
            dfts = pd.read_csv(filename[0], low_memory=False)
            dfts = dfts.rename(columns={'Weight': 'weight'})
            dfts = dfts.drop(columns=['Complexity', '30sTnterval', 'start_transition'])
            interval2 = intervals[i]
            dfts['Source'] = dfts['Source'].values + str(intervals[i])
            dfts['Target'] = dfts['Target'].values + str(intervals[i])
            dfts['temporal_connect'] = 0

            row1 = [dfts['ID'].iloc[0], dfts['ExpertLevel'].iloc[0]]
            row2 = [1, 0, 0, 0, 0, 1]

            for aoi in aoi_lst:
                row = row1 + [aoi + '_' + interval1, aoi + '_' + interval2] + row2
                dft.loc[len(dft)] = row

            interval1 = interval2

            dft = pd.concat([dft, dfts], axis=0)

    return dft


def load_node_dataset(node_names, identifier):
    node_names = node_names[node_names['IDint'] == identifier]
    intervals = np.sort(node_names['Interval'].unique().astype(int)).astype(str)
    for i in range(len(node_names)):
        filename = node_names[node_names['Interval'] == intervals[i]]['name'].values

        if i == 0:
            dfn = pd.read_csv(filename[0], low_memory=False)
            dfn['Node'] = dfn['Node'].values + str(intervals[i])
            dfn = dfn.drop(columns=['Complexity', '30sInterval', 'duration_start'])
        if i != 0:
            dfns = pd.read_csv(filename[0], low_memory=False)
            dfns['Node'] = dfns['Node'].values + str(intervals[i])
            dfns = dfns.drop(columns=['Complexity', '30sInterval', 'duration_start'])

            dfn = pd.concat([dfn, dfns], axis=0)

    return dfn


def add_idint_names(id_lst):
    split = id_lst['ID'].str.split('_', expand=True)
    id_lst['IDint'] = split.iloc[:, 0].str.split('D', expand=True).iloc[:, 1].astype(int)
    id_lst['Interval'] = split.iloc[:, 1]
    return id_lst


class Datasets:
    def __init__(self, trans_names, node_names, identifier, project_path, single_intervals):
        if not single_intervals:
            self.dft = load_transition_dataset(trans_names, identifier)
            self.dfn = load_node_dataset(node_names, identifier)
            self.dft = self.dft.rename(columns={'Weight': 'weight'})

        if single_intervals:
            self.dft = pd.read_csv(trans_names, low_memory=False)
            self.dft = self.dft.rename(columns={'Weight': 'weight'})
            self.dfn = pd.read_csv(node_names, low_memory=False)

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

    def get_y(self, target):
        if target == 'complexity':
            return int(self.dft['Complexity'].iloc[0])
        if target == 'expertise':
            return int(self.dft['ExpertLevel'].iloc[0])

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

    def add_node_attributes_to_networkx(self, structural_variables):
        if structural_variables is not None:
            # merge node attributes and structura variables
            self.dfn = self.dfn.merge(structural_variables, on='Node', how='left')

        self.dfn.index = self.dfn['Node']
        self.dfn = self.dfn.drop(columns=['Node'])
        self.dfn = self.dfn.transpose()
        dict = self.dfn.to_dict()
        nx.set_node_attributes(self.G, dict)

        G = self.G
        print(' ')

    def calculate_structural_variables(self, structural_variable_names):
        df_lst = list()
        if 'degree_centrality' in structural_variable_names:
            # Add graph structural variables as node features
            degree_centrality = nx.degree_centrality(self.G)
            degree_centrality = pd.DataFrame.from_dict(degree_centrality, orient='index')
            degree_centrality = degree_centrality.reset_index()
            df_lst.append(degree_centrality)

        node_clique_number_lst = list()
        if 'node_clique_number' in structural_variable_names:
            node_clique_number = nx.node_clique_number(self.G)
            node_clique_number = pd.DataFrame.from_dict(node_clique_number, orient='index')
            node_clique_number = node_clique_number.reset_index()
            df_lst.append(node_clique_number)

        if len(structural_variable_names) == 0:
            return None

        if len(structural_variable_names) == 1:
            df = df_lst[0]
            df.columns = ['Node'] + structural_variable_names
            return df_lst[0]

        if len(structural_variable_names) > 1:
            df = df_lst[0]
            for i in range(1, len(df_lst)):
                df = df.merge(df_lst[i], on='index')
                df.reset_index()
            df.columns = ['Node'] + structural_variable_names
            return df

    def create_graph(self, edge_attribute_names, node_attribute_names, structural_variable_names, target,
                     fill_graph_with_zero_nodes, single_intervals):
        if target in ['complexity', 'expertise']:
            y = self.get_y(target=target)

        edge_attr = ['weight'] + edge_attribute_names

        aoi_lst = ['1A', '1B', '2A', '2B', '3A', '3B', '4A', '4B', '5A', '5B', '6A', '6B',
                   '7A', '7B', '8A', '8B', '9A', '9B', 'PB']

        # Modify adjacency data frame
        dftt = pd.concat([self.dft[['Source', 'Target']], self.dft[edge_attr]], axis=1)
        self.dft = dftt.groupby(['Source', 'Target']).sum().reset_index()

        if fill_graph_with_zero_nodes:
            # Fill missing edges with zero
            for source in aoi_lst:
                for target in aoi_lst:
                    if source != target:
                        if np.logical_and(source not in self.dft['Source'].values,
                                          target not in self.dft['Target'].values):
                            zeros = [0] * (len(edge_attribute_names) + 1)
                            self.dft.loc[len(self.dft)] = [source, target] + zeros

        # Modify node dataframe
        dfnn = pd.concat([self.dfn[['Node']], self.dfn[node_attribute_names]], axis=1)
        self.dfn = dfnn.groupby('Node').agg({'AOI_duration': np.sum, 'clicked': np.sum, 'pupil_diameter': np.mean,
                                             'controller_duration_on_aoi': np.sum, 'distance_to_aoi': np.mean,
                                             'seating_row_aoi': np.mean, 'seating_loc_aoi': np.mean,
                                             'duration_time_until_first_fixation': np.min, 'active_disruption': np.max,
                                             'passive_disruption': np.max}).reset_index()


        if target == 'disruptions':
            y = np.where(self.dfn['active_disruption'] == 1, 1,
                         np.where(self.dfn['passive_disruption'] == 1, 1, 0))  # HERE
            self.dfn = self.dfn.drop(columns=['active_disruption', 'passive_disruption'])  # HERE
            node_attribute_names.remove('active_disruption')  # HERE
            node_attribute_names.remove('passive_disruption')  # HERE

        if target == 'clicked':
            y = np.where(self.dfn['clicked'] == 1, 1, 0)  # HERE
            self.dfn = self.dfn.drop(columns=['clicked'])  # HERE
            node_attribute_names.remove('clicked')  # HERE

        if fill_graph_with_zero_nodes:
            # Fill missing nodes with zero
            for aoi in aoi_lst:
                if aoi not in self.dfn['Node'].values:
                    zeros = [0] * len(node_attribute_names)
                    self.dfn.loc[len(self.dfn)] = [aoi] + zeros
                    self.dft.loc[len(self.dft), 'duration_time_until_first_fixation'] = 30

        # Create graph
        self.create_undirected_graph_with_networkx(edge_attr)

        if not single_intervals:
            # Add graph structural variables as node features
            structural_variables = self.calculate_structural_variables(structural_variable_names)

            # Add further node attributes
            self.add_node_attributes_to_networkx(structural_variables)
            node_attribute_names = node_attribute_names + structural_variable_names

        G = self.get_networkx_graph()
        # Create Data format to save
        data = from_networkx(self.G, group_node_attrs=node_attribute_names, group_edge_attrs=edge_attr)
        data['y'] = y  # Add target
        edge_weights = data['edge_attr'][:, 0]
        data.edge_weight = edge_weights

        return data


def create_graphs(project_path, target, edge_attribute_names, node_attribute_names, structural_variables,
                  fill_graph_with_zero_nodes, single_intervals):
    trans_lst = locate_transition_data(project_path)
    node_lst = locate_node_data(project_path)

    save_path = project_path + '\\data\\graphs\\{}\\'.format(target)
    delete_files_in_directory(save_path)

    print('Create graph Datasets:')

    if not single_intervals:
        trans_lst = add_idint_names(trans_lst)
        node_lst = add_idint_names(node_lst)

        for id_int in trans_lst['IDint'].unique():
            data = Datasets(trans_lst, node_lst, id_int, project_path, single_intervals)

            if len(data.get_data()) != 0:
                if structural_variables:
                    structural_variable_names = ['degree_centrality', 'node_clique_number']
                if not structural_variables:
                    structural_variable_names = []
                graph = data.create_graph(edge_attribute_names, node_attribute_names, structural_variable_names,
                                          target=target, fill_graph_with_zero_nodes=fill_graph_with_zero_nodes,
                                          single_intervals=single_intervals)
                torch.save(graph, save_path + 'ID_' + id_int + '.pt')

    if single_intervals:
        for i in range(len(trans_lst)):
            trans_name = trans_lst['name'].iloc[i]
            print(trans_name)
            node_name = node_lst['name'].iloc[i]
            print(node_name)
            identifier = trans_lst['ID'].iloc[i]
            print('ID {}'.format(identifier))

            data = Datasets(trans_name, node_name, identifier, project_path, single_intervals)

            name = trans_name.split('.')[0].rsplit('_', 1)[0]
            if len(data.get_data()) != 0:
                if structural_variables:
                    structural_variable_names = ['degree_centrality', 'node_clique_number']
                if not structural_variables:
                    structural_variable_names = []
                graph = data.create_graph(edge_attribute_names, node_attribute_names, structural_variable_names,
                                          target=target, fill_graph_with_zero_nodes=fill_graph_with_zero_nodes,
                                          single_intervals=single_intervals)
                torch.save(graph, save_path + name + '.pt')
