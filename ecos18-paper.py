
# coding: utf-8

# In[84]:


import pandas as pd
import sklearn
import time
import numpy as np
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
get_ipython().run_line_magic('pylab', 'inline')
#%%

t1 = time.time()
print('Loading database ...')
df = pd.read_hdf('database/all_data_comp.h5','table')
print('Time to load database:', time.time()-t1)
#%%

# Variable names.
import var_names
d = var_names.d


# # Variables to use. First only RPM.
# 
# | test_no | rpm | FRP | exh_T | TC_rpm |
# |--|--|--|--|--|
# | 1  | x | _ | _ | _ |
# | 2  | x | x | _ | _ |
# | 3  | x | x | x | _ |
# | 4  | x | x | x | x |
# | 5  | _ | x | _ | _ |
# | 6  | _ | x | x | _ |
# | 7  | _ | x | x | x |
# | 8  | x | x | _ | _ |
# | 9  | _ | _ | x | _ |
# | 10 | _ | _ | x | x |
# | 11 | x | _ | x | _ |
# | 12 | _ | _ | _ | x |
# | 13 | x | _ | _ | _ |
# | 14 | x | _ | _ | _ |
# | 15 | x | _ | _ | _ |
# | 16 | x | _ | _ | _ |
# 
# 

# In[2]:


# Check number of combinations, just to be sure.



features =  ['rpm',
             'frp',
             'exh_T',
             'TC_rpm']



import itertools

def list_of_combs(arr):
    """returns a list of all subsets of a list"""
    
    combs = []
    for i in range(1, len(arr)+1):
        listing = [list(x) for x in itertools.combinations(arr, i)]
        combs.extend(listing)
    return combs

# Not used, does not produce a good list..
#
#for l in range(1, len(features)+1):
#    for subset in itertools.combinations(features, l):
#        print(subset)
#        
#comb = list()
#
#for i in range(0,len(features)):
#    for a in itertools.combinations(features,i+1):
#        comb.append(a)

combinations = list_of_combs(features)
for i in range(len(combinations)):
    print(combinations[i])

print('\nNumber of combinations:',len(combinations))


# In[90]:


d['ae1_rpm']


# In[109]:



# The combinations are done manually in an Excel workbook, it was too tedious to make. This way it is easier but
# might not be the best way... it works...


feat = pd.read_excel('training_setup.xlsx',index_col='test_no')

# Create a list of features for each test. A list which will contain a list of features for each row. This list
# will be used for the training.

test_features = list()


for i in range(30):
    
    tmp_l = list()
    
    if feat.iloc[i][0] == 1:
        #print('ett')
        for j in range(1,5):
            tmp_l.append(d[feat.iloc[i][j]])
            print('Test:',i+1,feat.iloc[i][j])
        
    elif feat.iloc[i][0] == 2:
        #print('två')
        for j in range(1,5):
            tmp_l.append(d[feat.iloc[i][j]])
            print('Test:',i+1,feat.iloc[i][j])
        for j in range(5,9):
            tmp_l.append(d[feat.iloc[i][j]])
            print('Test:',i+1,feat.iloc[i][j])
        
    elif feat.iloc[i][0] == 3:
        for j in range(1,5):
            tmp_l.append(d[feat.iloc[i][j]])
            print('Test:',i+1,feat.iloc[i][j])
        for j in range(5,9):
            tmp_l.append(d[feat.iloc[i][j]])
            print('Test:',i+1,feat.iloc[i][j])
        for j in range(9,13):
            tmp_l.append(d[feat.iloc[i][j]])
            print('Test:',i+1,feat.iloc[i][j])
        
    elif feat.iloc[i][0] == 4:
        for j in range(1,5):
            tmp_l.append(d[feat.iloc[i][j]])
            print('Test:',i+1,feat.iloc[i][j])
        for j in range(5,9):
            tmp_l.append(d[feat.iloc[i][j]])
            print('Test:',i+1,feat.iloc[i][j])
        for j in range(9,13):
            tmp_l.append(d[feat.iloc[i][j]])
            print('Test:',i+1,feat.iloc[i][j])
        for j in range(13,17):
            tmp_l.append(d[feat.iloc[i][j]])
            print('Test:',i+1,feat.iloc[i][j])
    
    test_features.append(tmp_l)

# And then at last add the corresponding predictor to each test set.


for i in range(len(test_features)):
    if 'AE1' in test_features[i][0]:
        test_features[i].append(d['fo_booster_13'])
    if 'AE2' in test_features[i][0]:
        test_features[i].append(d['fo_booster_24'])


# In[112]:


test_features[15]


# In[101]:


df[test_features[0]]


# In[114]:


test_no = 0
df_test = df[test_features[test_no]].dropna()
df_test.drop(df_test.columns[len(df_test.columns)-1],axis=1)


# In[119]:


len(df_test.columns)


# In[104]:


test_features[0]


# In[129]:


# Train model
# 

from sklearn.externals import joblib
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split

gen = 1
cores = -1

# plotting setup

n1 = 100
sample_n = 200

seed = 42 # This is to get a reproduceability.




for test_no in range(len(test_features)):
    
    print('Test number: ', test_no, '\n')
    df_test = df[test_features[test_no]].dropna()
    X = np.array(df_test.drop(df_test.columns[len(df_test.columns)-1],axis=1))
    y = np.array(df_test[df_test.columns[len(df_test.columns)-1]])

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=seed)
    tpot = TPOTRegressor(generations=gen, population_size=50, verbosity=2, n_jobs=cores)
    tpot.fit(X_train, y_train.reshape(-1,))
    
    features_readable = list()
    for t in range(len(test_features[test_no])):
        features_readable.append(d[test_features[test_no][t]])

    x = linspace(n1+1,n1+sample_n,sample_n)
    fig, ax = plt.subplots()
    fig.set_size_inches(22, 13)
    plt.plot(x,tpot.predict(X_test)[n1:n1+sample_n])
    plt.plot(x,y_test[n1:n1+sample_n])

    ax.set(xlabel='sample no', ylabel='FO flow m3/h',
           title='Training number:' + str(test_no) + '\nFeatures: \n '+ str(features_readable))
    ax.grid()

    fig.savefig("results/test_no_" + str(test_no) + ".png")



# In[218]:


# Train linear  models
# 

from sklearn.externals import joblib
#from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split

### Train a linear model just for comparison.

gen = 1
cores = -1

results = list()
cols = ['test_no','model','CV-score','CV_perc']

# plotting setup

n1 = 2000
sample_n = 200

seed = 42 # This is to get reproduce.


for test_no in range(len(test_features)):
    
    print('Test number, linear model: ', test_no, '\n')
    df_test = df[test_features[test_no]].dropna()
    X = np.array(df_test.drop(df_test.columns[len(df_test.columns)-1],axis=1))
    y = np.array(df_test[df_test.columns[len(df_test.columns)-1]])

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=seed)
    
    m_linear = sklearn.linear_model.LinearRegression(n_jobs=-1)
    m_linear.fit(X_train, y_train.reshape(-1,))

    score = m_linear.score(X_test,y_test)
    score_perc = m_linear.score(X_test,y_test)/max(y_test)
    print('Score: ',score )
    print('Score in % of max: ',score_perc )
    
    results.append([test_no,'linear',score,score_perc])
    
    features_readable = list()
    for t in range(len(test_features[test_no])):
        features_readable.append(d[test_features[test_no][t]])

    x = linspace(n1+1,n1+sample_n,sample_n)
    
    sizes = [[10,6],[12,8],[14,10]]
    
    for P in sizes:
        fig, ax = plt.subplots()
        fig.set_size_inches(P)
        plt.plot(x,m_linear.predict(X)[n1:n1+sample_n])
        plt.plot(x,y[n1:n1+sample_n])

        ax.set(xlabel='sample no', ylabel='FO flow m3/h',
               title='Sklearn Linear model. Training number:' + str(test_no) + '\nFeatures: \n '+ str(features_readable))
        ax.grid()

        fig.savefig("results/run_180205/linear_test_no_" + str(test_no) + str(P) + ".png")
    
    # tpot model
    
    print('Test number, TPOT model: ', test_no, '\n')
    

    m_tpot = TPOTRegressor(generations=gen, population_size=50, verbosity=2, n_jobs=cores)
    m_tpot.fit(X_train, y_train.reshape(-1,))

    score = m_tpot.score(X_test,y_test)
    score_perc = m_tpot.score(X_test,y_test)/max(y_test)
    print('Score: ',score )
    print('Score in % of max: ',score_perc )
    
    results.append([test_no,'tpot',score,score_perc])
    
    features_readable = list()
    for t in range(len(test_features[test_no])):
        features_readable.append(d[test_features[test_no][t]])

    x = linspace(n1+1,n1+sample_n,sample_n)
    
    sizes = [[10,6],[12,8],[14,10]]
    for P in sizes:
        fig, ax = plt.subplots()
        fig.set_size_inches(P)
        plt.plot(x,m_linear.predict(X)[n1:n1+sample_n])
        plt.plot(x,y[n1:n1+sample_n])

        ax.set(xlabel='sample no', ylabel='FO flow m3/h',
               title='Sklearn TPOT model. Training number:' + str(test_no) + '\nFeatures: \n '+ str(features_readable))
        ax.grid()

        fig.savefig("results/run_180205/tpot_test_no_" + str(test_no) + str(P) + ".png")
    
    

results = pd.DataFrame(results, columns=cols)
results.to_excel('results/run_180205/test_run.xlsx')


# In[151]:


print('Test number, linear model: ', test_no, '\n')
test_no = 0
seed = 42

df_test = df[test_features[test_no]].dropna()
X = np.array(df_test.drop(df_test.columns[len(df_test.columns)-1],axis=1))
y = np.array(df_test[df_test.columns[len(df_test.columns)-1]])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=seed)

m_linear = sklearn.linear_model.LinearRegression(n_jobs=-1)
m_linear.fit(X_train, y_train.reshape(-1,))

print(m_linear.score(X_test,y_test))

#plt.plot(x,m_linear.predict(X)[n1:n1+sample_n])
#plt.plot(x,y[n1:n1+sample_n])
plt.plot(y)


# In[203]:


results


# In[190]:


import csv
with open('logg.csv','wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(results)


# In[200]:


a = pd.DataFrame(results,columns=results[0])


# In[201]:


a


# In[128]:


features_readable = list()
for t in range(len(test_features[test_no])):
    features_readable.append(d[test_features[test_no][t]])
    
x = linspace(n1+1,n1+sample_n,sample_n)
fig, ax = plt.subplots()
fig.set_size_inches(22, 13)
plt.plot(x,tpot.predict(X_test)[n1:n1+sample_n])
plt.plot(x,y_test[n1:n1+sample_n])

ax.set(xlabel='sample no', ylabel='FO flow m3/h',
       title='Training number:' + str(test_no) + '\nFeatures: \n '+ str(features_readable))
ax.grid()

fig.savefig("results/test_no_" + str(test_no) + ".png")


# In[122]:


df_test[df_test.columns[len(df_test.columns)-1]]


# In[71]:



n1 = 100
sample_n = 200
x = linspace(n1+1,n1+sample_n,sample_n)

fig, ax = plt.subplots()

fig.set_size_inches(22, 13)

plt.plot(x,tpot.predict(X_test)[n1:n1+sample_n])
plt.plot(x,y_test[n1:n1+sample_n])

ax.set(xlabel='sample no', ylabel='FO flow Eng1_3 RPM Predictor',
       title='TPOT regression 2 generations snapshot\n FO flow Birka 180123')
ax.grid()

fig.savefig("tpot_eng13_rpm_2gen.png")
plt.show()


# In[29]:


plt.scatter(tpot.predict(X_test)[100:200],y_test[100:200])


# In[57]:


linspace(100,200)


# In[2]:


#More predictors

eng_13 = [d['ae1_rpm'],
          d['ae3_rpm'],
          d['me1_rpm'],
          d['me3_rpm'],
          d['ae1_frp'],
          d['ae3_frp'],
          d['me1_frp'],
          d['me3_frp'],
          d['ae1_exh_T'],
          d['ae3_exh_T'],
          d['me1_exh_T'],
          d['me3_exh_T'],
          d['fo_booster_13']
          ]

eng_24 = [d['ae2_rpm'],
          d['ae4_rpm'],
          d['me2_rpm'],
          d['me4_rpm'],
          d['ae2_frp'],
          d['ae4_frp'],
          d['me2_frp'],
          d['me4_frp'],
          d['ae2_exh_T'],
          d['ae4_exh_T'],
          d['me2_exh_T'],
          d['me4_exh_T'],
          d['fo_booster_24']
          ]

#%%

####
#### Training the first set with only rpm predictor
####

print('Features and predictions for training 2:\n\nEngine 1_3:')

for n in eng_13:
    print('- ',d[n])
print('\nEngine 2_4:')
for n in eng_24:
    print('- ',d[n])

print('\nDate: ',time.strftime('%y%m%d'))
print('Time: ',time.strftime('%H:%M:%S'))


# In[37]:



# We do not have the full time extent for the dataset, so we have to filter out.

date_begin = '2014-02-01'
date_end = '2014-12-16'


# The dataset is not complete overlapping in time with data from both the mass-flow meters and the
# the rest of the data. So we have to manually filter out the time interval which we are interested in.

    
#df_1_3[labels_1_3].plot()
#df_2_4[labels_2_4].plot()

df_1_3 = df[eng_13][date_begin:date_end].dropna()
df_2_4 = df[eng_24][date_begin:date_end].dropna()

for n in list(df_1_3):
    df_1_3[n][(df_1_3[n] < 0)] = 0

for n in list(df_2_4):
    df_2_4[n][(df_2_4[n] < 0)] = 0


# In[38]:


# Train model
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split

gen = 10
cores = -1

X = np.array(df_1_3.drop(labels=d['fo_booster_13'],axis=1))
y = np.array(df_1_3[d['fo_booster_13']])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)
tpot = TPOTRegressor(generations=gen, population_size=50, verbosity=2, n_jobs=cores)
tpot.fit(X_train, y_train.reshape(-1,))


# In[41]:



n1 = 500
sample_n = 200
x = linspace(n1+1,n1+sample_n,sample_n)

fig, ax = plt.subplots()

fig.set_size_inches(10, 5)

plt.plot(x,tpot.predict(X_test)[n1:n1+sample_n],label='Predicted value')
plt.plot(x,y_test[n1:n1+sample_n],label='Truth value')

ax.set(xlabel='sample no', ylabel='FO flow Eng1_3 RPM, Exh_T, FRP Predictor',
       title='Randomized test data\nTPOT regression 10 generations snapshot\n FO flow Birka 180125')
ax.grid()

plt.legend()

fig.savefig("tpot_eng13_rpm_exh_t_frp_pred_10gen.png",dpi=300)
plt.show()


# In[50]:



#Not randomized data. During a full day of operation.

n1 = 0
sample_n = 96
x = linspace(n1+1,n1+sample_n,sample_n)

fig, ax = plt.subplots()

fig.set_size_inches(10, 5)

plt.plot(x,tpot.predict(X)[n1:n1+sample_n],label='Predicted value')
plt.plot(x,y[n1:n1+sample_n],label='Truth value')

ax.set(xlabel='sample no', ylabel='FO flow Eng1_3 RPM, Exh_T, FRP Predictor',
       title='One day operation Eng1_3 \n TPOT regression 10 generations snapshot \n FO flow Birka 180125')
ax.grid()

plt.xlim(0)
plt.ylim(0)

plt.legend()

fig.savefig("tpot_eng13_op-data-rpm_exh_t_frp_pred_10gen.png",dpi=400)
plt.show()


# In[43]:



n1 = 500
sample_n = 200
x = linspace(n1+1,n1+sample_n,sample_n)

fig, ax = plt.subplots()

fig.set_size_inches(10, 10)

plt.scatter(tpot.predict(X_test)[n1:n1+sample_n],y_test[n1:n1+sample_n],color='r',marker='x')
plt.plot(tpot.predict(X_test)[n1:n1+sample_n],tpot.predict(X_test)[n1:n1+sample_n])

#plt.scatter(n1,n1+sample_n)

ax.set(xlabel='predicted value', ylabel='truth value',
       title='Randomized test data\n TPOT regression 10 generations snapshot\n FO flow Birka 180125')
ax.grid()

#plt.legend()
plt.xlim(0)
plt.ylim(0)

fig.savefig("tpot_eng13_scatter_rpm_exh_t_frp_pred_10gen.png",dpi=300)
plt.show()


# In[45]:




fig, ax = plt.subplots()

fig.set_size_inches(10, 10)

plt.scatter(tpot.predict(X),y,color='r',marker='x')
plt.plot(tpot.predict(X),tpot.predict(X))

#plt.scatter(n1,n1+sample_n)

ax.set(xlabel='predicted value', ylabel='truth value',
       title='One year data\n TPOT regression 10 generations snapshot\n FO flow Birka 180125')
ax.grid()

#plt.legend()
plt.xlim(0)
plt.ylim(0)

fig.savefig("tpot_eng13_all_scatter_rpm_exh_t_frp_pred_10gen.png",dpi=300)
plt.show()


# In[47]:


tpot.export('gen10_eng13_180125.py')


# In[65]:


tpot.score(X_test,y_test)


# In[66]:


print('MSE in % ', abs(tpot.score(X_test,y_test)/max(y)*100))


# In[67]:


from sklearn.metrics import mean_squared_error
mean_squared_error(tpot.predict(X_test),y_test)


# In[ ]:



for n in list(df_1_3):
    df_1_3[n][(df_1_3[n] < 0)] = 0

for n in list(df_2_4):
    df_2_4[n][(df_2_4[n] < 0)] = 0
    
df_1_3[labels_1_3].plot()
df_2_4[labels_2_4].plot()

