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

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances

import nltk
import string

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



def generate_hashtags(input_files=[],input_dir=''): #change here
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



def extract_hash_tags(s):
    x=set(part[1:] for part in s.split() if part.startswith('#'))
    xx=re.sub(r'[{}.]','',str(x))
    return xx









if __name__ == "__main__":
  # argument parse
  parser = argparse.ArgumentParser()
  #parser.add_argument("--lda_top_20_keywords_file", help = "top 20 keywords for each topic file",required=True)
  #parser.add_argument("--dec_keyword_file", help = "top dec keywords",required=False)
  #parser.add_argument("--word_vectors_voc_file", help = "file of look up table of word embeddings and dec/lda keywords",required=True)
  parser.add_argument("--query_input_files", help = "directory contains files to start query on",required=True)
  #parser.add_argument("--start_window",type=int, help = "start query from after this window",required=True)
  #parser.add_argument("--n", type=int,help = "number of keywords to use for query",required=True)
  #parser.add_argument("--query_output_directory",help="directory to save query result files",required=True)
  parser.add_argument("--output_file",help="file to save tweet count for each topic",required=True)
  parser.add_argument("--output_image",help="file to save tweet count for each topic",required=True)
  parser.add_argument("--topics",type=int, help = "number of topic",required=True)
  #parser.add_argument("--keyword", help = "hashtag_keyword",required=True)
  args = parser.parse_args()

  input_files=get_files(file_folder=args.query_input_files)

  input_topici = [x for x in input_files if "topic"+str(args.topics) in x]

  data=generate_hashtags(input_topici,args.query_input_files)

  t_h=[]
  for i in data:
      t_h.append(extract_hash_tags(i))



  l = list(filter(lambda x: x != 'set()', t_h))
  ll = [''.join(c for c in s if c not in string.punctuation) for s in l]

  data_hash=ll

   
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


  vectorizer = TfidfVectorizer(stop_words='english')
  X = vectorizer.fit_transform(data_hash)
  '''
  true_k = 10
  model = KMeans(n_clusters=true_k, init='k-means++')
  model.fit(X)

  fff=open(args.output_file+'_top5_topics'+str(args.topics)+'.txt','w')

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

  fff=open(args.output_file+'_top10_topics'+str(args.topics)+'.txt','w')

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
      
      cost.append(KM.inertia_)      
  
  # plot the cost against K values
  plt.figure(figsize = (10,6))
  plt.plot(range(1, 10), cost, color ='g', linewidth ='3') 
  plt.xlabel("Value of K") 
  plt.ylabel("Sqaured Error (Cost)")
  #plt.legend()
  plt.grid(which='minor')  
  #plt.show() # clear the plot 
  plt.savefig(args.output_image+'_10k_topics'+str(args.topics)+'.jpg')  

 
  # the point of the elbow is the # most optimal value for choosing k  
  cost =[]
  for i in range(1, 20):
      KM = KMeans(n_clusters = i)
      KM.fit(X)

      # calculates squared error
      # for the clustered points
      cost.append(KM.inertia_)

  # plot the cost against K values
  plt.figure(figsize = (10,6))
  plt.plot(range(1, 20), cost, color ='g', linewidth ='3')
  plt.xlabel("Value of K")
  plt.ylabel("Sqaured Error (Cost)")
  #plt.legend()
  plt.grid(which='minor') 
  #plt.show() # clear the plot
  plt.savefig(args.output_image+'_20k_topics'+str(args.topics)+'.jpg')

#  plt.savefig('lda_20_hash_20k.jpg')

# the point of the elbow is the
# most optimal value for choosing k
   '''


from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from yellowbrick.cluster import KElbowVisualizer

# Generate synthetic dataset with 8 random clusters

# Instantiate the clustering model and visualizer
model = KMeans()
visualizer = KElbowVisualizer(model, k=(2,15))

visualizer.fit(X)        # Fit the data to the visualizer
visualizer.show(args.output_image+'_15k_topics'+str(args.topics)+'.png')        # Finalize and render the figure
#plt.savefig(args.output_image+'_20k_topics'+str(args.topics)+'.jpg')
