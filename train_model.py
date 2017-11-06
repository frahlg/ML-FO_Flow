import pandas as pd
import sklearn
import time
import numpy as np
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

def train_tpot(name,X,y,gen,cores):

    test_name = str('gen_' + str(gen) + name + '_' + time.strftime('%y%m%d'))

    print('Training with TPOT .... ', test_name)
    t1 = time.time()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)
    tpot = TPOTRegressor(generations=gen, population_size=50, verbosity=2, n_jobs=cores)
    tpot.fit(X_train, y_train.reshape(-1,))

    print(tpot.score(X_test,y_test))
    t2 = time.time()
    delta_time = t2-t1
    print('Time to train...:', delta_time)


    print('Saving the model ...')
    tpot.export('trained_models/' + test_name + '.py')
    joblib.dump(tpot.fitted_pipeline_,'trained_models/' + test_name + '.pk1')
    print(test_name, ' saved ... ')
