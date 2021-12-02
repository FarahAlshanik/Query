
import pandas as pd
f=pd.read_csv('dd.txt',names=['time'])

j=0
data=[]

for i in f['time']:
    #print(i.split(' ')[0])
    y=float(i.split(' ')[0])
    x=int(i.split(' ')[1])
    data.append([x,y])   

s=sorted(data)

dis=open('sorted_d.txt','w')
for i in s:
	dis.write('%s ' % i)
	dis.write('\n')
dis.close()

