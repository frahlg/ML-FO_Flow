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

##
# Features and target for Eng 1/3


#%%

gen = 50
cores=-1 # use all of them


test_name = str('gen_' + str(gen) + '_rpm_predictor_'+time.strftime('%y%m%d'))
print('Training... test ...', test_name)

eng_13 = [d['ae1_rpm'],
          d['ae3_rpm'],
          d['me1_rpm'],
          d['me3_rpm'],
          d['fo_booster_13']
          ]

eng_24 = [d['ae2_rpm'],
          d['ae4_rpm'],
          d['me2_rpm'],
          d['me4_rpm'],
          d['fo_booster_24']
          ]

#%%

####
#### Training the first set with only rpm predictor
####

print('Features and predictions for training 1:\n\nEngine 1_3:')

for n in eng_13:
    print('- ',d[n])
print('\nEngine 2_4:')
for n in eng_24:
    print('- ',d[n])

print('\nDate: ',time.strftime('%y%m%d'))
print('Time: ',time.strftime('%H:%M:%S'))
#%%
# Drop Nan from the DataFrame.

# Create training arrays, X_13 is the features for engine pair 1 and 3

df_1_3 = df[eng_13].dropna()
X = np.array(df_1_3.drop(labels=d['fo_booster_13'],axis=1))
y = np.array(df_1_3[d['fo_booster_13']])


print('Training with TPOT engine pair 1....', test_name)
t1 = time.time()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)
tpot = TPOTRegressor(generations=gen, population_size=50, verbosity=2, n_jobs=cores)
tpot.fit(X_train, y_train.reshape(-1,))

print(tpot.score(X_test,y_test))
t2 = time.time()
delta_time = t2-t1
print('Time to train...:', delta_time)

#%%

print('Saving the model ...')
tpot.export('trained_models/eng13_' + test_name + '.py')
joblib.dump(tpot.fitted_pipeline_,'trained_models/eng13_' + test_name + '.pk1')
print(test_name, ' saved ... ')

#%%

# X_24 features for engine 2, 4

df_2_4 = df[eng_24].dropna()
X = np.array(df_2_4.drop(labels=d['fo_booster_24'],axis=1))
y = np.array(df_2_4[d['fo_booster_24']])

print('Training with TPOT engine pair 2....', test_name)
t1 = time.time()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)
tpot = TPOTRegressor(generations=gen, population_size=50, verbosity=2, n_jobs=cores)
tpot.fit(X_train, y_train.reshape(-1,))

print(tpot.score(X_test,y_test))
t2 = time.time()
delta_time = t2-t1
print('Time to train...:', delta_time)

#%%

print('Saving the model ...')
tpot.export('trained_models/eng24_' + test_name + '.py')
joblib.dump(tpot.fitted_pipeline_,'trained_models/eng24_' + test_name + '.pk1')
print(test_name, ' saved ... ')
