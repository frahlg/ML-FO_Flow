# Train a TPOT model

import pandas as pd
import sklearn
import time
import numpy as np
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib


t1 = time.time()
print('Loading database ...')
df = pd.read_hdf('../database/all_data_comp.h5','table')
print('Time to load database:', time.time()-t1)


# Features and target for Eng 1/3

labels_1_3 = ['AE1 FUEL RACK POSIT:1742:mm:Average:900',
                'AE3 FUEL RACK POSIT:3742:mm:Average:900',
                'AE1 ENG SPEED:1745:RPM:Average:900',
                'AE3 ENG SPEED:3745:RPM:Average:900',
                'ME1 FUEL RACK POSIT:10005:%:Average:900',
                'ME3 FUEL RACK POSIT:30005:%:Average:900',
                'ME1 ENGINE SPEED:1364:rpm:Average:900',
                'ME3 ENGINE SPEED:3364:rpm:Average:900',
                'FO BOOST 1 CONSUMPT:6165:m3/h:Average:900']

# Features and target for Eng 2/4

labels_2_4 = ['AE2 FUEL RACK POSIT:2742:mm:Average:900',
                'AE4 FUEL RACK POSIT:4742:mm:Average:900',
                'AE2 ENG SPEED:2745:RPM:Average:900',
                'AE4 ENG SPEED:4745:RPM:Average:900',
                'ME2 FUEL RACK POSIT:20005:%:Average:900',
                'ME4 FUEL RACK POSIT:40005:%:Average:900',
                'ME2 ENGINE SPEED:2364:rpm:Average:900',
                'ME4 ENGINE SPEED:4364:rpm:Average:900',
             'FO BOOST 2 CONSUMPT:6166:m3/h:Average:900']


print('Features and predictions for training 1', labels_1_3, labels_2_4)

# Drop Nan from the DataFrame.

df_1_3 = df[labels_1_3].dropna()
df_2_4 = df[labels_2_4].dropna()

# Create training arrays, X_13 is the features for engine pair 1 and 3

X_13 = np.array(df_1_3.drop(labels='FO BOOST 1 CONSUMPT:6165:m3/h:Average:900',axis=1))
y_13 = np.array(df_1_3['FO BOOST 1 CONSUMPT:6165:m3/h:Average:900'])

# X_24 features for engine 2, 4

X_24 = np.array(df_2_4.drop(labels='FO BOOST 2 CONSUMPT:6166:m3/h:Average:900',axis=1))
y_24 = np.array(df_2_4['FO BOOST 2 CONSUMPT:6166:m3/h:Average:900'])


print('Training with TPOT engine pair 1....')

#  modeling with tpot
# Number of generations, should be boosted if we want better results. 3 gens take ~ 30min on my macbook...
gen = 1
cores=-1 # use all of them

X_train_13, X_test_13, y_train_13, y_test_13 = train_test_split(X_13, y_13, train_size=0.75, test_size=0.25)
tpot = TPOTRegressor(generations=gen, population_size=50, verbosity=2, n_jobs=cores)
tpot.fit(X_train_13, y_train_13.reshape(-1,))

print(tpot.score(X_test_13,y_test_13))

tpot.export('train_tpot_1_3.py')
joblib.dump(tpot,'train_tpot_1_3.pk1')
