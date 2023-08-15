import numpy as np
import pandas as pd



def add_expertise_levels(dff):
    import os
    os.chdir(r'V:\VirATeC\data\VirATeC_NSR\2_questionnaire')
    df_q = pd.read_csv(r"pre_post.csv", sep=";", index_col=False, decimal=',', encoding='ISO-8859-1')
    df_q.insert(1, 'Expert?', np.where(df_q['years'] > 0, 1, 0))
    df_q = df_q[['ID', 'Expert?']]

    dff = dff.merge(df_q, on='ID')

    return dff

'''
# n_columns = len(mat_lst[0])
# columns = list(map(str, np.arange(n_columns)))
columns.append('condition')
columns.append('ID')

df_input = pd.DataFrame(columns=columns)
for i in range(len(mat_lst)):
    mat = mat_lst[i]
    cond = cond_lst[i]
    ID = ID_lst[i]

    values = list(mat['Weight'].values)
    values.append(cond)
    values.append(ID)
    df_input.loc[len(df_input) ,:] = values

df_input = add_expertise_levels(df_input)

df_input.to_csv(project_path + '//data//transitions.csv' ,index=False)
'''