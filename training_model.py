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
d = var_names.dict_


# Check if variables exist in the dictonary..

for names in d:
    if d[names] in list(df):
        #print(var_names[names])
        pass
    else:
        print('*** VAR MISSING *** ', var_names[names], ' *** VAR MISSING ***')


# Features and target for Eng 1/3

eng_13 = [d['ae1_frp'],
          d['ae3_frp'],
          d['ae1_cac_P'],
          d['ae3_cac_P'],
          d['ae1_cac_ca'],
          d['ae3_cac_ca'],
          d['ae1_exh'],
          d['ae3_exh'],
          d['ae1_fo_P'],
          d['ae3_fo_P'],
          d['ae1_rpm'],
          d['ae3_rpm'],
          d['me1_frp'],
          d['me3_frp'],
          d['me1_ca_T'],
          d['me3_ca_T'],
          d['me1_cac_T'],
          d['me3_cac_T'],
          d['me1_exh_T'],
          d['me3_exh_T'],
          d['me1_rpm'],
          d['me3_rpm'],
          d['fo_booster_13']
          ]

# Features and target for Eng 2/4

eng_24 = [d['ae2_frp'],
          d['ae4_frp'],
          d['ae2_cac_P'],
          d['ae4_cac_P'],
          d['ae2_cac_ca'],
          d['ae4_cac_ca'],
          d['ae2_exh'],
          d['ae4_exh'],
          d['ae2_fo_P'],
          d['ae4_fo_P'],
          d['ae2_rpm'],
          d['ae4_rpm'],
          d['me2_frp'],
          d['me4_frp'],
          d['me2_ca_T'],
          d['me4_ca_T'],
          d['me2_cac_T'],
          d['me4_cac_T'],
          d['me2_exh_T'],
          d['me4_exh_T'],
          d['me2_rpm'],
          d['me4_rpm'],
          d['fo_booster_24']
          ]



#%%

# Feature selection

print('Which variables..')
gen = int(float(input('How many generations?')))
cores = int(float(input('How many cores? (-1 all cores')))

#%%

# If we want an interactive list to choose variables from.
#
#
# import inquirer
# questions = [
#     inquirer.Checkbox('Variables engine 1/3',
#                     message='Choose variables for training',
#                     choices=eng_13)
#             ]
# answ = inquirer.prompt(questions)
# print(answ)


#%%



print('Features and predictions for training 1', labels_1_3, labels_2_4)
#%%
# Drop Nan from the DataFrame.

df_1_3 = df[labels_1_3].dropna()
df_2_4 = df[labels_2_4].dropna()

# Create training arrays, X_13 is the features for engine pair 1 and 3

X_13 = np.array(df_1_3.drop(labels='FO BOOST 1 CONSUMPT:6165:m3/h:Average:900',axis=1))
y_13 = np.array(df_1_3['FO BOOST 1 CONSUMPT:6165:m3/h:Average:900'])

# X_24 features for engine 2, 4

X_24 = np.array(df_2_4.drop(labels='FO BOOST 2 CONSUMPT:6166:m3/h:Average:900',axis=1))
y_24 = np.array(df_2_4['FO BOOST 2 CONSUMPT:6166:m3/h:Average:900'])


#%%
print('Training with TPOT engine pair 1....')
t1 = time.time()

#  modeling with tpot
# Number of generations, should be boosted if we want better results. 3 gens take ~ 30min on my macbook...
gen = 50
cores=-1 # use all of them

X_train_13, X_test_13, y_train_13, y_test_13 = train_test_split(X_13, y_13, train_size=0.75, test_size=0.25)
tpot_13 = TPOTRegressor(generations=gen, population_size=50, verbosity=2, n_jobs=cores)
tpot_13.fit(X_train_13, y_train_13.reshape(-1,))

print(tpot_13.score(X_test_13,y_test_13))
t2 = time.time()
delta_time = t2-t1
print('Time to train...:', delta_time)

#%%

print('Saving the model ...')

tpot_13.export('train_tpot_13_gen50.py')
joblib.dump(tpot_13.fitted_pipeline_,'train_tpot_13_gen50.pk1')

print('Model saved ... ')


#%%

print('Training with TPOT engine pair 2....')
t1 = time.time()

#  modeling with tpot
# Number of generations, should be boosted if we want better results. 3 gens take ~ 30min on my macbook...

X_train_24, X_test_24, y_train_24, y_test_24 = train_test_split(X_24, y_24, train_size=0.75, test_size=0.25)
tpot_24 = TPOTRegressor(generations=gen, population_size=50, verbosity=2, n_jobs=cores)
tpot_24.fit(X_train_24, y_train_24.reshape(-1,))

print(tpot_24.score(X_test_24,y_test_24))
t2 = time.time()
delta_time = t2-t1
print('Time to train...:', delta_time)

#%%

print('Saving the model ...')

tpot_24.export('train_tpot_24_gen50.py')
joblib.dump(tpot_24.fitted_pipeline_,'train_tpot_24_gen50.pk1')

print('Model saved ... ')
