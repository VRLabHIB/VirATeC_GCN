import glob
import os
import pandas as pd
import numpy as np
import torch

## for from networkx function:
from collections import defaultdict
from typing import Any,  List, Optional,  Union

from torch import Tensor
import torch_geometric


def locate_raw_data():
    os.chdir(r'V:\VirATeC\data\VirATeC_NSR\0_raw_data')
    lst = glob.glob("*Main.csv")

    df_lst = pd.DataFrame({'name': lst})
    df_lst['ID'] = df_lst['name'].str.split('_', expand=True).iloc[:, 0]

    return df_lst


def locate_processed_data():
    os.chdir(r'V:\VirATeC\data\VirATeC_NSR\1_full_sessions')
    lst = glob.glob("*.csv")

    df_lst = pd.DataFrame({'name': lst})
    df_lst['ID'] = df_lst['name'].str.split('.', expand=True).iloc[:, 0]

    return df_lst


def locate_transition_data(project_path):
    data_path = project_path + '\\data\\nodes_and_transitions\\'
    os.chdir(data_path)
    lst = glob.glob("*_trans.csv")

    df_lst = pd.DataFrame({'name': lst})
    df_lst['ID'] = df_lst['name'].str.split('.', expand=True).iloc[:, 0]

    return df_lst

def locate_node_data(project_path):
    data_path = project_path + '\\data\\nodes_and_transitions\\'
    os.chdir(data_path)
    lst = glob.glob("*_node.csv")

    df_lst = pd.DataFrame({'name': lst})
    df_lst['ID'] = df_lst['name'].str.split('.', expand=True).iloc[:, 0]

    return df_lst


def locate_graph_data(project_path):
    data_path = project_path + '\\data\\graphs\\'
    os.chdir(data_path)
    lst = glob.glob("*.pt")

    df_lst = pd.DataFrame({'name': lst})
    df_lst['ID'] = df_lst['name'].str.split('.', expand=True).iloc[:, 0]

    return df_lst


def calculate_time_diff(df):
    """

    Parameters
    ----------
    df : pandas dataframe
        raw, cleaned dataframe.
    time : string
        variable that stores the time changes of an experiment [sec].

    Returns
    -------
    time_diff : list
        conveys time difference between following steps of an experiment.

    """

    # create list with start time of the experiment
    time_diff = np.array([0])

    # create two vectors
    t0 = np.array(df['Time'][:-1].tolist())
    t1 = np.array(df['Time'][1:].tolist())

    # vectors subtraction
    diff = np.subtract(t1, t0)

    time_diff = np.append(time_diff, diff)

    return time_diff


def calculate_object_stats(df):
    df_stats = pd.DataFrame(df['GazeTargetObject'].value_counts() / len(df) * 100)
    df_stats = df_stats.rename(columns={'GazeTargetObject': 'PercentLookedAt'})
    df_stats.insert(0, 'GazeTargetObject', df_stats.index)
    df_stats = df_stats.reset_index(drop=True)

    lst = ['PresentationBoard', 'none']
    line = pd.DataFrame({"GazeTargetObject": 'Child_total',
                         "PercentLookedAt": len(df[~df['GazeTargetObject'].isin(lst)]) / len(df) * 100}, index=[3])
    df_stats = pd.concat([df_stats.iloc[:2], line, df_stats.iloc[2:]]).reset_index(drop=True)

    return df_stats


def add_expertise_levels(dff):
    import os
    os.chdir(r'V:\VirATeC\data\VirATeC_NSR\2_questionnaire')
    df_q = pd.read_csv(r"pre_post.csv", sep=";", index_col=False, decimal=',', encoding='ISO-8859-1')
    df_q.insert(1, 'Expert?', np.where(df_q['years'] > 0, 1, 0))
    df_q = df_q[['ID', 'Expert?']]

    id_lst = list()
    for i in range(len(dff)):
        identifier = dff['ID'].iloc[i]
        identifier = identifier[2] + identifier[3] + identifier[4]
        identifier = int(identifier)
        id_lst.append(identifier)
    dff = dff.drop(columns=['ID'])
    dff.insert(0, 'ID', id_lst)

    dff = dff.merge(df_q, on='ID')
    expert = dff['Expert?'].values
    dff = dff.drop(columns=['Expert?'])
    dff.insert(4, 'Expert?', expert)

    return dff


def from_networkx(
    G: Any,
    group_node_attrs: Optional[Union[List[str], all]] = None,
    group_edge_attrs: Optional[Union[List[str], all]] = None,
) -> 'torch_geometric.data.Data':
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
        group_node_attrs (List[str] or all, optional): The node attributes to
            be concatenated and added to :obj:`data.x`. (default: :obj:`None`)
        group_edge_attrs (List[str] or all, optional): The edge attributes to
            be concatenated and added to :obj:`data.edge_attr`.
            (default: :obj:`None`)

    .. note::

        All :attr:`group_node_attrs` and :attr:`group_edge_attrs` values must
        be numeric.

    Examples:

        >>> edge_index = torch.tensor([
        ...     [0, 1, 1, 2, 2, 3],
        ...     [1, 0, 2, 1, 3, 2],
        ... ])
        >>> data = Data(edge_index=edge_index, num_nodes=4)
        >>> g = to_networkx(data)
        >>> # A `Data` object is returned
        >>> from_networkx(g)
        Data(edge_index=[2, 6], num_nodes=4)
    """
    import networkx as nx

    from torch_geometric.data import Data

    mapping = dict(zip(G.nodes(), range(G.number_of_nodes())))
    edge_index = torch.empty((2, G.number_of_edges()), dtype=torch.long)
    for i, (src, dst) in enumerate(G.edges()):
        edge_index[0, i] = mapping[src]
        edge_index[1, i] = mapping[dst]

    data = defaultdict(list)

    if G.number_of_nodes() > 0:
        node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
    else:
        node_attrs = {}

    if G.number_of_edges() > 0:
        edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())
    else:
        edge_attrs = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        if set(feat_dict.keys()) != set(node_attrs):
            raise ValueError('Not all nodes contain the same attributes')
        for key, value in feat_dict.items():
            data[str(key)].append(value)

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        if set(feat_dict.keys()) != set(edge_attrs):
            raise ValueError('Not all edges contain the same attributes')
        for key, value in feat_dict.items():
            key = f'edge_{key}' if key in node_attrs else key
            data[str(key)].append(value)

    for key, value in G.graph.items():
        if key == 'node_default' or key == 'edge_default':
            continue  # Do not load default attributes.
        key = f'graph_{key}' if key in node_attrs else key
        data[str(key)] = value

    for key, value in data.items():
        if isinstance(value, (tuple, list)) and isinstance(value[0], Tensor):
            data[key] = torch.stack(value, dim=0)
        else:
            try:
                data[key] = torch.tensor(value)
            except (ValueError, TypeError):
                pass

    data['edge_index'] = edge_index.view(2, -1)
    data = Data.from_dict(data)

    if group_node_attrs is all:
        group_node_attrs = list(node_attrs)
    if group_node_attrs is not None:
        xs = []
        for key in group_node_attrs:
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.x = torch.cat(xs, dim=-1)

    if group_edge_attrs is all:
        group_edge_attrs = list(edge_attrs)
    if group_edge_attrs is not None:
        xs = []
        for key in group_edge_attrs:
            key = f'edge_{key}' if key in node_attrs else key
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.edge_attr = torch.cat(xs, dim=-1)

    if data.x is None and data.pos is None:
        data.num_nodes = G.number_of_nodes()

    return data

