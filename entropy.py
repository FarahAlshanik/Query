import pandas as pd
import string
import csv
import datetime
import pandas
import csv
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
 
import pylab
import matplotlib.pyplot as plt
plt.switch_backend('agg')
 
import matplotlib.pyplot as plt  
import random

import numpy as np
import argparse
from os import listdir
from os.path import isfile, join
import sys
import dec_text_plda
import dec_graph
import re
import numpy as np
import pandas as pd
from pprint import pprint
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.ldamulticore import LdaMulticore
# Plotting tools
# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import csv
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
from collections import Counter




def get_files(file_folder='', file_format='file_%d.csv'):
        """
        Return all interval files in a given folder.
        Usage:
                #files = get_files('/zfs/dicelab/farah/query_exp/query-construction2/Baltimore/interval/int/', 'file_%d.csv')
                #files=get_files('/zfs/dicelab/farah/query_exp/query-construction2/Baltimore/interval/queryoutput/lda_int15/', 'file_%d.csv')
                # files = ['hourly_intervals/file_1.csv' ... 'hourly_inverals/file_216.csv']

        :param file_folder: where your files are
        :param file_format: how your files are named
        :returns files: ordered list of files to process.
        """
        if len(file_folder):
                if file_folder[-1] != '/':
                        file_folder += '/'

        num_files = len([f for f in listdir(file_folder) if isfile(join(file_folder, f))])

        files = []
        for i in range(0, num_files):
                files.append(file_folder + file_format % (i + 1))
                print(files[-1])

        return files


input_files =  get_files(file_folder="/zfs/dicelab/farah/break_police_15M")


j=0
file=[]
for i in input_files:
    file.append(pd.read_csv(i,encoding='latin-1',sep=',', 
                  names=["time", "abstract", "id"]))


import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
stemmer = PorterStemmer()
import os
def review_to_words(raw_review):
    
    # 1. Remove non-letters        
    #letters_only = re.sub("[^a-zA-Z0-9]", " ", raw_review) 
    
    # 2. Convert to lower case, split into individual words
    words = raw_review.lower().split()
    
    # 3. Remove Stopwords. In Python, searching a set is much faster than searching a list, so convert the stop words to a set
    
    stops = set(stopwords.words("english"))                  
    stops.add('#')
    # 4. Remove stop words
    meaningful_words = [w for w in words if not w in stops]  #returns a list 

    # 5. Stem words. Need to define porter stemmer above
    singles = [stemmer.stem(word) for word in meaningful_words]
    
    # 6. Join the words back into one string separated by space, and return the result.
    return( " ".join( singles ))

# build a corpus for the word2vec model

def removeHashTag(text):
        """
        Remove hashtag form givne set of words.
        """
        cleanedText = []
        for word in text:
                word = word.replace("#","")
                word = word.replace("@","")
                word = word.replace(":","")
                word = word.replace(",","")
                word = word.replace("!","")
                word = word.replace(".","")
                
                cleanedText.append(word)
        return ("".join( cleanedText ))

processed_file=[]
for i in file:
    processed_file.append([review_to_words(text) for text in i['abstract']])

processed_files=[]
for i in processed_file:
    processed_files.append([removeHashTag(text) for text in i])

entropy=[]
for i, _ in enumerate(processed_files):
    entropy.append("entropy"+str(i+1))

import collections
cnt = collections.Counter()
s=[]
k=0 
dict = {}
for i, documents in enumerate(processed_files):
    dict[entropy[k]]=[] 
    cnt = collections.Counter()
    for doc in documents:
        cnt[doc] += 1
    dict[entropy[k]].append(cnt)
    k+=1
         
         #cnt[doc] += 1
#print(cnt.items())        
#print(cnt)#return the freq of word in corpus


def prob(word,blob,bloblist,cnt2):
    return (cnt2[0][word]/len(bloblist))

def entropy_fun(word,blob,bloblist,cnt2):
    return -(np.log2( prob(word,blob,bloblist,cnt2))*prob(word,blob,bloblist,cnt2))

k=0 
dict_entropy = {}
for i, documents in enumerate(processed_files):
    dict_entropy[entropy[k]]=[]  
    for doc in documents:
        #print(len(documents))
        dict_entropy[entropy[k]].append([k,entropy_fun(doc,doc,documents,dict[entropy[k]])]) # k is the file number mean all these result belongs to k file
    k+=1
         
         #cnt[doc] += 1
#print(cnt.items())        
#print(cnt)#return the freq of word in corpus

s=0
k=0
ent=[]
for i in range(len(dict_entropy)):
    s=0
    for j in dict_entropy[entropy[k]]:
        s+=j[1]
    ent.append([k,s/len(dict_entropy[entropy[k]])])    
    k+=1

x_data=[]
y_data=[]
for i in ent:
    x_data.append(i[0])
    y_data.append(i[1])
    
   
 

 

 
 
# From here the plotting starts

plt.scatter(x_data, y_data, c='r', label='data')
#plt.plot(x_func, y_func, label='$f(x) = 0.388 x^2$')
plt.xlabel('Time Window')
plt.ylabel('Tweet Entropy')
plt.title('Tweet Entropy over Time Window for Police Tweets')
plt.legend()
#plt.show()
plt.savefig('tweet_entropy_police.jpg')
