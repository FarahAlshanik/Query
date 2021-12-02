import pylab
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import matplotlib.pyplot as plt
import random

import pandas as pd


#data=pd.read_csv('tweet_entropy_key_value.txt',encoding='latin-1',sep=',', 
#                  names=["time"])

#data=pd.read_csv('new_entropy_combine_interval_tweet_new.txt',encoding='latin-1',sep=',',
 #                 names=["time"])
#data=pd.read_csv('new_entropy_combine_interval_words_new_2_new.txt',encoding='latin-1',sep=',',
 #                 names=["time"])


#data=pd.read_csv('new_entropy_combine_interval_tweet_new_2.txt',encoding='latin-1',sep=',',
 #                 names=["time"])


#data=pd.read_csv('new_entropy_combine_interval_words_new_2.txt',encoding='latin-1',sep=',',
 #                 names=["time"])

#data=pd.read_csv('new_entropy_combine_interval_words_sliding_window.txt',encoding='latin-1',sep=',',
 #                 names=["time"])

#data=pd.read_csv('similarity_centroid_combined_sliding_3.txt',encoding='latin-1',sep=',',
                  #names=["time"])


data=pd.read_csv('entropy_words_sliding_window.txt',encoding='latin-1',sep=',',
                  names=["time"])


#print(data.head())

#print(data["time"][2].split("]")[0])

x_data=[]
y_data=[]
j=0


for i in data['time']:
    y_data.append(i)
    x_data.append(j)
    j+=1
'''

for i in data['time']:
    y_data.append(float(data["time"][j].split("]")[0].lstrip()))
    x_data.append(j)
    j+=1
'''    
print(y_data[0])
print(y_data[1])

plt.plot(x_data,y_data)
plt.xlabel('Time_Interval')
plt.ylabel('Entropy')
plt.title('Keywords_Entropy')


plt.savefig('eks.jpg')
'''
entropy_file_num=[]
for i in range(len(y_data)):
    entropy_file_num.append([x_data[i],y_data[i]])

ss=sorted(entropy_file_num,key = lambda x: x[1])
for i in ss:
        print(i)
#print(ss)

with open('hig_low_new_similarity_sliding_window.txt', 'w') as filehandle:
    for listitem in ss:
        filehandle.write('%s\n' % listitem)

'''
