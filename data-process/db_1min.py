
import pandas as pd
import glob

csv_data_path = 'csv/'
xls_data_path = 'original/'
database_path = 'database/'


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
