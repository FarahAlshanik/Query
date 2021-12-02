import argparse
from os import listdir
from os.path import isfile, join
import sys
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
from gensim.models import word2vec
import argparse
from os import listdir
from os.path import isfile, join
import sys
import re
import os

from nltk.corpus import stopwords
from nltk.stem.porter import *
stemmer = PorterStemmer()
import os.path
import gensim
# load the word2vec algorithm from the gensim library
from gensim.models import word2vec
import dec_main
import dec_text
from nltk.stem import WordNetLemmatizer


def removeStopwords(text=[]):
        """
        Remove stopwords from a list of text.
        """
        stops = set(stopwords.words("english"))

        text = [x for x in text if x.lower() not in stops]
        text = [x for x in text if len(x) > 1]
#       text = " ".join(text) #only this extra
#       print(text)
        return text

def stemText(text=[]):
        """
        Stem text.
        """
        wordnet_lemmatizer = WordNetLemmatizer()
        stemmed = []
        #text=[]
        for word in text:
                stemmed.append(wordnet_lemmatizer.lemmatize(word))
        #       text.append( " ".join(stemmed))
#       print(stemmed)
        return stemmed

def removeHashTag(text=[]):
        """
        Remove hashtag form givne set of words.
        """
        cleanedText = []
        for word in text:
                word = word.replace("#","")
                cleanedText.append(word)
        return cleanedText






def preprocessKeywords(text=[]):
        """
        Process all tokens.
        :param text: word tokens representing a document (ie tweet).
        :param stopwords: stopwords to filter on.
        :returns text: cleaned text
        """
        tweet=getWords(text)
        text = removeHashTag(tweet)
        text = stemText(tweet)
        text = removeStopwords(tweet)
        return( " ".join( text ))


def getWords(text):
        """
        Remove some speical characters and hyperlinks from text.
        :param text: string representing a tweet
        :returns words: cleaned tokens.
        """
        text = re.sub("<.*?>","",text.lower())
        text = re.sub(r"http\S+", "", text)
        words = re.compile(r'[^A-Z^a-z]+').split(text)
        return words




import argparse
from os import listdir
import argparse
from os import listdir

#parser = argparse.ArgumentParser()
#args = parser.parse_args()
#input_files = os.listdir(args.input_dir)
#input_files =  get_files(file_folder="/zfs/dicelab/farah/break_police_15M")

#parser = argparse.ArgumentParser()
#parser.add_argument("--input_dir",help="input directory of query result files to count hashtags", required=True)
#args = parser.parse_args()
#input_files = os.listdir(args.input_dir)

#input_files =  get_files(file_folder="/zfs/dicelab/farah/query_exp/query-construction2/sample")

##input_dirc='/zfs/dicelab/farah/query_exp/query-construction2/sample/'
#input_dirc='/zfs/dicelab/farah/break_police_15M/'


parser = argparse.ArgumentParser()

parser.add_argument("--input_dir",help="input directory of query result files to count hashtags", required=True)

args = parser.parse_args()

input_files = os.listdir(args.input_dir)


split_numbers=[]
for i in input_files:
    split_numbers.append(int(i.split('_')[1].split('.')[0]))
    #print(split_numbers)
    sorted_num=sorted(split_numbers)
    files=[]
    for i in sorted_num:
        files.append("file"+"_"+str(i)+".csv")

full_path=[]
for interval, f in enumerate(files):
    print("Processing data from file " + f)
    inputf0 = os.path.join(f)
    full_path.append(args.input_dir+f)
    #print(inputf0)
    #file=pd.read_csv(inputf0,encoding='latin-1',sep=',', names=["time", "abstract", "id"])

print(full_path)





j=0
file=[]
for i in full_path:
    #print(i)
    file.append(pd.read_csv(i,encoding='latin-1',sep=',', 
                  names=["time", "abstract", "id"]))
    #print(i)
    #j+=1


print(len(file))
print(file[0].head())


import pandas as pd
import numpy as np
import re
import nltk

processed_files=[]
for i in file:
    processed_files.append([preprocessKeywords(text) for text in i['abstract']])


print(processed_files[0])


entropy=[]
for i, _ in enumerate(processed_files):
    entropy.append("entropy"+str(i))

import collections

#cnt_times = collections.Counter()
dict = {} # to stor the counter for time interval
k=0
len_files=[]# to have the number of documents in each 2 time interval
s=0
#count_times=[]
for i in range(len(processed_files)):
    #s=cnt_times
    #count_times.append(s)
    dict[entropy[k]]=collections.Counter()
    #cnt_times_new = collections.Counter()
    if(k==0):
        s+=len(processed_files[i])
    for doc in (processed_files[i]):
         dict[entropy[k]][doc]+=1
    if(k!=0):
        
        dict[entropy[k]]=dict[entropy[k]]+dict[entropy[k-1]]
        s+=len(processed_files[i])
        len_files.append(s) 
    else:
        len_files.append(len(processed_files[i]))
                

        
    #print(i)
    #print(cnt_times)
    k+=1
   
    #print(s)
    #for k,v in  cnt_times.most_common():
     #   f.write(k+":"+str(v)+" " )
    #f.write("\n")
    
    

import numpy as np
def prob(word,blob,bloblist,cnt2):
    return (cnt2[word]/(bloblist))

def entropy_fun(word,blob,bloblist,cnt2):
    return (np.log2( prob(word,blob,bloblist,cnt2))*prob(word,blob,bloblist,cnt2))


e=[]
s=0
entropy_times=[]
for i in range (len(entropy)):
    s=0
    for doc in dict[entropy[i]]:
        entropy_val=entropy_fun(doc,doc,len_files[i],dict[entropy[i]])
        e.append([i,entropy_val])
        s+=entropy_val
    entropy_times.append(-s)
    print(i)

#print(entropy_times)


dis = open('new_entropy_combine_interval_topic2.txt', 'w')
for item in entropy_times:
  dis.write("%s\n" % item)
dis.close()


import pylab
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import matplotlib.pyplot as plt
import random

import pandas as pd




x_data=[]
y_data=[]

j=0


for i in entropy_times:
    y_data.append(i)
    x_data.append(j)
    j+=1

plt.figure(figsize=(20,10))


plt.plot(x_data,y_data)
plt.savefig('tweet_entropy_cum_topic2.jpg')


