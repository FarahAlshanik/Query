import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from os import listdir
from os.path import isfile, join
import math
from sklearn.cluster import KMeans
import re
import matplotlib.pyplot as plt
plt.switch_backend('agg')


import pylab
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import matplotlib.pyplot as plt
import random




import csv, subprocess
import time
import os
import sys
import argparse
from os import listdir

def get_files(file_folder=''):
        if len(file_folder):
                if file_folder[-1] != '/':
                        file_folder += '/'

        num_files = len([f for f in listdir(file_folder) if isfile(join(file_folder, f))])

        files = [f for f in listdir(file_folder) if isfile(join(file_folder, f))]
        #print(files)
        #for i in range(0, num_files):
         #       files.append(file_folder + file_format % (i + 1))
          #      print(files[-1])

        return files



#we should change this 
input_files=get_files(file_folder='/scratch1/falshan/dec_results/query_results_some_intervals_lda/lda20_781')
input_topici = [x for x in input_files if "topic"+str(0) in x]



def generate_hashtags(input_files=[],input_dir='/scratch1/falshan/dec_results/query_results_some_intervals_lda/lda20_781'): #change here
    #count number of hashtags from query results
        t=[]
        for interval, f in enumerate(input_files):
                #print("Processing data from file " + f)
                inputf0 = open(os.path.join(input_dir,f),'r')
                inputf = csv.reader(inputf0, delimiter = ',')
                for tweet in inputf:
                        t.append(tweet[1].lower())
                        #content = tweet[1]
                        
                inputf0.close()
        return t
        #sorted_hashtags = sorted(hashtags.items(), key = lambda kv: kv[1],reverse=True)
        #with open(output_file,'w') as outputf:
         #       for k,v in sorted_hashtags:
          #              outputf.write(k+" "+str(v)+"\n")


data=generate_hashtags(input_topici)

def extract_hash_tags(s):
    x=set(part[1:] for part in s.split() if part.startswith('#'))
    xx=re.sub(r'[{}.]','',str(x))
    return xx


t_h=[]
for i in data:
    t_h.append(extract_hash_tags(i))


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cluster import KMeans 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances

import nltk
import string



l = list(filter(lambda x: x != 'set()', t_h))
ll = [''.join(c for c in s if c not in string.punctuation) for s in l]

data_hash=ll

from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
   
ps = PorterStemmer() 
  
data_stem=[]  
for w in data_hash: 
    data_stem.append(ps.stem(w)) 

data_stem_no_acci=[]
for i in data_stem:
    data_stem_no_acci.append(re.sub(r'[^\x00-\x7F]+',' ', i))

l_no_asci = list(filter(lambda x: x != ' ', data_stem_no_acci))
l_no_asci_n = list(filter(lambda x: x != '', l_no_asci))

l_no_asci_n_2=[]
for i in l_no_asci_n:
    l_no_asci_n_2.append(i.strip())



data_hash=l_no_asci_n_2

#from pandas import DataFrame
#df = DataFrame (data_hash,columns=['hash_tags'])

#change here
#df.to_csv('lda_20_hash_2.csv',index=False)

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data_hash)

true_k = 10
model = KMeans(n_clusters=true_k, init='k-means++')
model.fit(X)

fff=open('lda_781int_top5.txt','w')

print("Top terms per cluster:")
print()
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    fff.write(" Cluster %d:" % i+"\n")
    print(" Cluster %d:" % i),
    for ind in order_centroids[i, :5]:
        print(' %s' % terms[ind], end=' '),
        fff.write(' %s' % terms[ind])
    fff.write('\n\n')    
    print()
    print()

print("\n")
fff.close() 





fff=open('lda_781int_top10.txt','w')

print("Top terms per cluster:")
print()
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    fff.write(" Cluster %d:" % i+"\n")
    print(" Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind], end=' '),
        fff.write(' %s' % terms[ind])
    fff.write('\n\n')
    print()
    print()










cost =[] 
for i in range(1, 10): 
    KM = KMeans(n_clusters = i) 
    KM.fit(X) 
      
    # calculates squared error 
    # for the clustered points 
    cost.append(KM.inertia_)      
  
# plot the cost against K values 
plt.plot(range(1, 10), cost, color ='g', linewidth ='3') 
plt.xlabel("Value of K") 
plt.ylabel("Sqaured Error (Cost)") 
#plt.show() # clear the plot 
plt.savefig('lda_20_hash_10k.jpg')  
 

 
# the point of the elbow is the  
# most optimal value for choosing k  




cost =[]
for i in range(1, 20):
    KM = KMeans(n_clusters = i)
    KM.fit(X)

    # calculates squared error
    # for the clustered points
    cost.append(KM.inertia_)

# plot the cost against K values
plt.plot(range(1, 20), cost, color ='g', linewidth ='3')
plt.xlabel("Value of K")
plt.ylabel("Sqaured Error (Cost)")
#plt.show() # clear the plot

plt.savefig('lda_20_hash_20k.jpg')

# the point of the elbow is the
# most optimal value for choosing k

