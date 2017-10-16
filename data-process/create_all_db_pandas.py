
# coding: utf-8

# # Script for merging all Excel-data from M/S Birka into one HDF5-database
# 
# The Pandas-library is used for both reading, writing and sorting the data.

# In[1]:


import pandas as pd
import glob

csv_data_path = 'csv/'
xls_data_path = 'original/'
database_path = 'database/'

xlsfiles = glob.glob(xls_data_path + '*.xls')
extra_xls = glob.glob(xls_data_path + '/extra/*.xlsx') # These are the additional data


# # Clean Excel files, make .CSV
# 
# Cleaning of the raw Excel files. Reading a .CSV file is much faster than a .XLS. So the first step is to clean and filter the data, and resave them into CSV files.
# 
# The data which is fetched from the ship is in exported with their database tool to Excel-97 files. As it wasn't possible to export more than a certain amount of rows (<65k) as well as the file being to large with to many columns, the data was
# exported in several batches. As this was done manually, by clicking thousands of small tick-boxes, this unfortunally meant that some data was overlapped both in time-series as well as duplicate data columns in different files. The data in the first 10 rows of the Excel-files
# contained meta-data about each data point which needed to be extracted and put togehter into one header. Some of the Excel-files also
# contained non ASCII-characters which also needed to be filtered out.
# 
# 
# 
# 

# In[6]:


####
####
# Clean up csv
# Make CSV-files and clean up headers.
####
####


# As there are non uni-code characters in the original headers file it needs be fixed..
# The following function was found here:
# http://stackoverflow.com/questions/20078816/replace-non-ascii-characters-with-a-single-space
# And replaces all non unicode chars with a space

def remove_non_ascii(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])

for i in range(len(xlsfiles)):
    df = pd.DataFrame()
    df2 = pd.DataFrame()

    print('Processing: '+str(xlsfiles[i].split('/')[-1].split('.')[0]))
    df = pd.read_excel(xlsfiles[i],index_col=0)
    df.index.name = 'Time'

    headers = list(df)
    headers_new = list()

    # And now extract the relevant meta-data in the first couple of rows.
    # Make a new list of headers in the file. Using ':' as a split.


    for head in headers:

        name = str(df[head].iloc[0])
        id_nr = str(head.split('.')[2].split(':')[1])
        unit = str(df[head].iloc[1])
        data_type = str(df[head].iloc[5])
        sample_interval = str(df[head].iloc[8])

        headers_new.append(str(name+':'+id_nr+':'+unit+':'+data_type+':'+sample_interval))

    for n in range(len(headers_new)):
        series = df[headers[n]].iloc[13:]
        df2[remove_non_ascii(headers_new[n])] = series


    # Save in .csv format.
    df2.to_csv(csv_data_path + xlsfiles[i].split('/')[-1].split('.')[0] + '.csv')
    #df2.to_excel(csv_data_path + xlsfiles[i].split('/')[-1].split('.')[0] + '.xls')

    # Clean up memory
    del df2
    del df
    print(str(i+1) + ' done of ' + str(len(xlsfiles)))


print('Batch one done...')


# In[ ]:



# No need for filtering out metadata from the extra files.

for i in range(len(extra_xls)):
    df = pd.DataFrame()

    print('Processing: '+str(extra_xls[i].split('/')[-1].split('.')[0]))
    df = pd.read_excel(xlsxfiles[i],index_col=0)
    df.index.name = 'Time'
    
    df.to_csv(csv_data_path + xlxsfiles[i].split('/')[-1].split('.')[0] + '.csv')
    del df
    print(str(i+1) + ' done of ' + str(len(xlsfiles)))

print('All done!')


# # Crate database for everything in 15-min frequency
# 
# All data is gathered into one master Pandas DataFrame. The append-function puts the the DataFrame in the bottom but keeps track
# of the columns. The resample-function is applied.

# In[ ]:


data_freq = '15min'

all_data=pd.DataFrame()
csvfiles = glob.glob(csv_data_path + '*.csv')

for i in range(len(csvfiles)):
    df = pd.DataFrame()
    df_out = pd.DataFrame()
    print('Processing: '+str(csvfiles[i].split('/')[-1].split('.')[0]))
    df = pd.read_csv(csvfiles[i],header=0,index_col=0,dtype='a')
    df.index = pd.to_datetime(df.index)
    
    print('Resampling: '+str(csvfiles[i].split('/')[-1].split('.')[0]))
    
    for n in range(len(list(df))):
        df_out[list(df)[n]] = pd.to_numeric(df[list(df)[n]],errors='ignore')
    
    print('Appending: '+str(csvfiles[i].split('/')[-1].split('.')[0]))
    all_data = all_data.append(df_out).resample(data_freq).mean()
    
    del df # Clean up memory
    del df_out # Clean up memory
    print(str(i+1) + ' done of ' + str(len(csvfiles)))


print('Saving database...\n')
all_data.to_hdf(database_path + 'all_data_comp.h5','table',complevel=9,complib='blosc') # compressed version

print('All done!')


# # Create a database 1 min interval
# 
# Some parts of the data is in 1-min interval. All data which are logged in 15-min interval will be filled with Nan
# between each 15-min point. This can be handled later on, but we don't loose accuracy.

# In[ ]:



data_freq = '1min'
all_data=pd.DataFrame()
csvfiles = glob.glob(csv_data_path + '*.csv')

for i in range(len(csvfiles)):
    df = pd.DataFrame()
    df_out = pd.DataFrame()
    print('Processing: '+str(csvfiles[i].split('/')[-1].split('.')[0]))
    df = pd.read_csv(csvfiles[i],header=0,index_col=0,dtype='a')
    df.index = pd.to_datetime(df.index)
    
    print('Resampling: '+str(csvfiles[i].split('/')[-1].split('.')[0]))
    
    for n in range(len(list(df))):
        df_out[list(df)[n]] = pd.to_numeric(df[list(df)[n]],errors='ignore')
    
    print('Appending: '+str(csvfiles[i].split('/')[-1].split('.')[0]))
    all_data = all_data.append(df_out).resample(data_freq).mean()
    
    del df # Clean up memory
    del df_out # Clean up memory
    print(str(i+1) + ' done of ' + str(len(csvfiles)))


print('Saving database...\n')
all_data.to_hdf(database_path + 'all_data_1min_comp.h5','table',complevel=9,complib='blosc') # compressed version

print('All done!')


