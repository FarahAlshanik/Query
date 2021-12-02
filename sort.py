import pandas as pd
data=pd.read_csv('entropy_tweet_sliding_window.txt',encoding='latin-1',sep=',',
                  names=["time"])


#print(data.head())

#print(data["time"][2].split("]")[0])

x_data=[]
y_data=[]
j=0

dd=[]
for i in data['time']:
    y_data.append(i)
    x_data.append(j)
    dd.append([i,j])
    j+=1


ddd=sorted(dd,key = lambda x: x[0])

dis = open('/zfs/dicelab/farah/query_exp/query-construction2/sorted_entr.txt', 'w')
for item in ddd:
  dis.write("%s\n" % item)
dis.close()

