
import pandas as pd

data=pd.read_csv('similarity_centroid_combined_sliding_3.txt',encoding='latin-1',sep=',',
                  names=["time"])


#print(data.head())

#print(data["time"][2].split("]")[0])

x_data=[]
y_data=[]
j=0


for i in data['time']:
    y_data.append(float(data["time"][j].split("]")[0].lstrip()))
    x_data.append(j)
    j+=1


s=0
k=0
avg=[]
for i in y_data:
    print(i)
    s+=i
    if(k==4):
        print("*********************************")
        avg.append(s/4)
        s=0
        k=0
    k+=1
       
print(y_data[0])
