# Train a TPOT model

import pandas as pd
import sklearn
import time
import numpy as np
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
#%%

t1 = time.time()
print('Loading database ...')
df = pd.read_hdf('database/all_data_comp.h5','table')
print('Time to load database:', time.time()-t1)
#%%

# Variable names.
import var_names
d = var_names.d

# Check if variables exist in the dictonary..
# for names in d:
#     if d[names] in list(df):
#         pass
#     else:
#         print('*** VAR MISSING *** ', d[names], ' *** VAR MISSING ***')


#%%

gen = 50
cores=-1 # -1 = use all of them, can be

import train_model as tm
#
# def train_tpot(name,X,y,gen,cores):
#


#%%

####
#### Training the first set with only rpm predictor
####
##
# Features and target for Eng 1/3


test_name = str('eng_13_exh_T_predictor_'+time.strftime('%y%m%d'))
features = [d['ae1_exh_T'],
          d['ae3_exh_T'],
          d['me1_exh_T'],
          d['me3_exh_T'],
          d['fo_booster_13']
          ]

print('Features and predictions for training...:\n')
for n in features:
    print('- ',d[n])

print('\nDate: ',time.strftime('%y%m%d'))
print('Time: ',time.strftime('%H:%M:%S'))

# Drop Nan from the DataFrame.

# Create training arrays, X_13 is the features for engine pair 1 and 3

df_train = df[features].dropna()
X = np.array(df_train.drop(labels=d['fo_booster_13'],axis=1))
y = np.array(df_train[d['fo_booster_13']])

tm.train_tpot(test_name,X,y,gen,cores)

##
##
##%%
## Training next engine pair.
##
# X_24 features for engine 2, 4


test_name = str('eng24_exh_T_predictor_'+time.strftime('%y%m%d'))
features = [d['ae2_exh_T'],
          d['ae4_exh_T'],
          d['me2_exh_T'],
          d['me4_exh_T'],
          d['fo_booster_24']
          ]

print('Features and predictions for training...:\n')
for n in features:
    print('- ',d[n])
print('\nDate: ',time.strftime('%y%m%d'))
print('Time: ',time.strftime('%H:%M:%S'))


df_train = df[features].dropna()
X = np.array(df_train.drop(labels=d['fo_booster_24'],axis=1))
y = np.array(df_train[d['fo_booster_24']])

tm.train_tpot(test_name,X,y,gen,cores)
