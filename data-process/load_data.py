
# coding: utf-8

# In[1]:


import pandas as pd
import time
import sklearn as sk 
from sklearn import preprocessing
get_ipython().magic('pylab inline')


# In[3]:


t1 = time.time()
print('Loading database ...')
df = pd.read_hdf('database/all_data_comp.h5','table')
print('Time to load database:', time.time()-t1)


# In[4]:


headers = list(df)
print('Number of headers:', len(headers))


# In[5]:


# write all headers to a CSV file

headers = open('database/headers.csv','w')
a = list(df)
for item in a:
    headers.write('\n' + item)
headers.close()


# In[82]:


# Check how much of the data is in there...

headers = list(df)
df['date'] = df.index

time_span = list()

for N in headers:
    df2 = df[['date', N]]
    start=df2.dropna().iloc[0].name.date()
    end=df2.dropna().iloc[-1].name.date()
    time_span.append([N, start, end])

spans_df = pd.DataFrame(time_span)
spans_df.to_excel('database/time_spans.xlsx')


# In[27]:


df.tail()


# In[20]:


df.describe()


# In[14]:


headers

