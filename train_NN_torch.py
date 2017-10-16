import pandas as pd
import torch
import sklearn as sk
from sklearn import preprocessing
import time

# Load the database, shouldn't take that long. This is all data!

t1 = time.time()
print('Loading database ...')
df = pd.read_hdf('database/all_data_comp.h5','table')
print('Time to load database:', time.time()-t1)

##%%
##
# The cost function
# The dataset is not complete overlapping in time with data from both the mass-flow meters and the
# the rest of the data. So we have to manually filter out the time interval which we are interested in.

date_begin = '2014-02-02'
date_end = '2014-12-15'

# Dict of var names we want to use.


var_names = {'ae1_frp':'AE1 FUEL RACK POSIT:1742:mm:Average:900',
             'ae2_frp':'AE2 FUEL RACK POSIT:2742:mm:Average:900',
             'ae3_frp':'AE3 FUEL RACK POSIT:3742:mm:Average:900',
             'ae4_frp':'AE4 FUEL RACK POSIT:4742:mm:Average:900',
             'me1_frp':'ME1 FUEL RACK POSIT:10005:%:Average:900',
             'me2_frp':'ME2 FUEL RACK POSIT:20005:%:Average:900',
             'me3_frp':'ME3 FUEL RACK POSIT:30005:%:Average:900',
             'me4_frp':'ME4 FUEL RACK POSIT:40005:%:Average:900'}

for names in var_names:
    print(var_names[names])

#%%




#Creating the Tensors:
#
#for name_ in var_names:
#    globals()[name_] = 3

fo_1_3_total = df['FO_day_engine_1_3']
fo_2_4_total = df['FO_day_engine_2_4']


#%%

# Define cost functions.

def day_cost(df,day):

    pred1_3 = np.sum((df[var_names['ae1_frp']][day] * w1 + df[var_names['ae1_frp']][day] * w2**2 + b)+
                     (df[var_names['ae3_frp']][day] * w1 + df[var_names['ae3_frp']][day] * w2**2 + b)+
                     (df[var_names['me1_frp']][day] * w1 + df[var_names['me1_frp']][day] * w2**2 + b)+
                     (df[var_names['me3_frp']][day] * w1 + df[var_names['me3_frp']][day] * w2**2 + b))


    #df['FO_day_engine_2_4'][day].max()
    pred = pred1_3
    target = df['FO_day_engine_1_3'][day].max()
    cost = np.square(pred - target)
    return pred, cost

def der_day_cost(x):
    return np.diff(x)
