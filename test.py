import pandas as pd
import numpy as np
from numpy.linalg import norm

import vg

append = False
if append:
    s1 = pd.Series()

    s1['ID'] = 'ID001'
    s1['movement'] = 528

    s2 = pd.Series()

    s2['ID'] = 'ID002'
    s2['movement'] = 587

    df = pd.DataFrame()

    df = df.append(s1, ignore_index=True)

    df = df.append(s2, ignore_index=True)

t = 3000
v=2

z = np.zeros(t)
o = np.ones(t)
z = np.array([np.zeros(t), np.ones(t)]).T
v = np.random.rand(t, v)

dot = np.einsum('ij,ij->i', z, v)
norm = norm(v, axis=1)
x = np.cross(z, v, axis=1)
sign = np.array(np.sign(np.cross(z, v, axis=1)))

angle = np.degrees(np.arccos(dot/norm))*sign

#c = np.dot(v1, base)/norm(v1)
#angle = np.degrees(np.arccos(np.clip(c, -1, 1)))

print('Test')


########
node_list = set(self.dft['Source'].to_list()).union(set(self.dft['Source'].to_list()))
node_attributes = np.random.rand(len(node_list), 2)

df_nodes = pd.DataFrame(node_attributes)

df_nodes.index = list(node_list)
df_nodes.columns = node_attribute_names
