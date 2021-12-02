
import pylab
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import matplotlib.pyplot as plt
import random


import pandas as pd

f=pd.read_fwf('jacard_par1.txt',names=['time'])

f2=pd.read_fwf('jacard_par2.txt',names=['time'])



j=0
data=[]
for i in f['time']:
    xx=f["time"][j].split("]")[0].lstrip()    
    y=(float(xx.split(' ')[0]))
    x=(int(xx.split(' ')[1]))
    data.append([x,y])
    j+=1

j=0
data2=[]
for i in f2['time']:
    xx=f2["time"][j].split("]")[0].lstrip()
    y=(float(xx.split(' ')[0]))
    x=(int(xx.split(' ')[1]))
    data2.append([x,y])
    j+=1



data_sore=sorted(data)

print(data_sore)


data2_sore=sorted(data2)

#print(data2_sore)

x_data=[]
y_data=[]

for i in data_sore:
    x_data.append(i[0])
    y_data.append(i[1])


for i in data2_sore:
    x_data.append(i[0])
    y_data.append(i[1])



print(x_data)
print(y_data)




plt.plot(x_data,y_data)
plt.savefig('par.jpg')

