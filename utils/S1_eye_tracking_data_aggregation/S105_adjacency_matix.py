import numpy as np
import pandas as pd
import networkx as nx


def build_matrices(dataframes):
    """
    dataframes: List of Dataframes. A Dataframe is a Pandas DataFrame object, containing the following variables:
        'time': Time point since the start of experiment
        'time_dur': Transition duration between consecutive OOIs
        'Source': OOI at the beginning of the gaze-shift
        'Target': Targeted OOI after gaze-shift
        'Weight': 1 if there is a gaze shift, 0 otherwise
    """

    mat_lst = list()
    cond_lst = list()
    ID_lst = list()

    for file in range(len(dataframes)):
        df_trans = dataframes[file]
        cond_lst.append(df_trans['condition'].iloc[0])
        ID_lst.append(df_trans['ID'].iloc[0])

        ooi_lst = ['1A', '1B', '2A', '2B', '3A', '3B', '4A', '4B', '5A', '5B', '6A', '6B',
                   '7A', '7B', '8A', '8B', '9A', '9B', 'PB']

        # ooi_lst = ['L', 'M', 'R', 'PB']
        source = list()
        target = list()
        weight = list()
        columns = list()
        for i in ooi_lst:
            for j in ooi_lst:
                columns.append('{}{}'.format(i, j))
                source.append(i)
                target.append(j)
                df_copy = df_trans[np.logical_and(df_trans['Target'] == i, df_trans['Source'] == j)]
                w_sum = np.sum(df_copy['Weight'].values)
                weight.append(w_sum)

        df_mat = pd.DataFrame({'Source': source, 'Target': target, 'Weight': weight})

        mat_lst.append(df_mat)

    return mat_lst, cond_lst, ID_lst, columns


# Function to split a string into a list of letters
def split(word):
    return [char for char in word]


def build_graphs(mat_lst, project_path, data_lst):
    graph_list = list()
    for df_mat in mat_lst:
        G = nx.from_pandas_edgelist(df_mat, source='Source', target='Target', edge_attr=['Weight'],
                                    create_using=nx.MultiDiGraph())
        graph_list.append(G)

    import pickle
    import os
    os.chdir(project_path + '//data//graphs//')
    for i in range(len(graph_list)):
        s = split(data_lst[i])
        name = s[0] + s[1] + s[2] + s[3]
        with open("{}.p".format(name), 'wb') as f:
            pickle.dump(graph_list[i], f)

    return graph_list


########################################################################################################################
########################################################################################################################
def create_matrices_and_graphs(project_path):
    import os
    import glob
    print('Start creating model dataframe')
    os.chdir(project_path + '//data//transitions//')
    data_lst = glob.glob("*.csv")
    print("number of object files: ", len(data_lst))
    dataframes = [pd.read_csv(f, sep=',', header=0) for f in data_lst]  # create list with all data frames loaded

    mat_lst, cond_lst, ID_lst, columns = build_matrices(dataframes)

    graph_list = build_graphs(mat_lst, project_path, data_lst)

    return mat_lst, graph_list, cond_lst, ID_lst
