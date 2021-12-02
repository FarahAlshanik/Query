import pandas as pd
#f=pd.read_csv('/zfs/dicelab/farah/query_exp/query-construction2/Baltimore/jac/emerging_windows_stop.txt',names=['jac'])

f=pd.read_csv('/zfs/dicelab/farah/Baltimore/jac/emerging_windows.txt',names=['jac'])


#print(f.head())


files=[]
for i in f['jac']:
#    print(i)
    files.append("file_"+str(i)+".csv")


#print(files)


input='/zfs/dicelab/farah/break_police_15M/'
ff=[]
for interval, d in enumerate(files):
        s=(d.split("_"))
        k=s[1].split(".")
        #print(k[0])
        #print(k[0])
        #if(int(k[0]) in f['jac']):
#            print(k[0])
        ff.append(input+d)
        #print("Processing data from interval %d" % (interval + 1))
    
print(ff)

import shutil, os
#files = ['file1.txt', 'file2.txt', 'file3.txt']


for f in ff:
    shutil.copy(f, '/zfs/dicelab/farah/Baltimore/lda_files/')

