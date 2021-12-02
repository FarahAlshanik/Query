
import collections
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
import os

parser = argparse.ArgumentParser()

parser.add_argument("--input_dir",help="input directory of query result files to count hashtags", required=True)
parser.add_argument("--window",help="sliding window for entropy", required=True)

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


print(files)

print("****************************************************************************")

full_path=[]
for interval, f in enumerate(files):
    print("Processing data from file " + f)
    inputf0 = os.path.join(f)
    full_path.append(args.input_dir+f)
    #print(inputf0)
    #file=pd.read_csv(inputf0,encoding='latin-1',sep=',', names=["time", "abstract", "id"])

print(full_path[2])

print("******************************************************************************")

w=int(args.window)
print(w)



j=0
file=[]
for i in full_path:
    #print(i)
    file.append(pd.read_csv(i,encoding='latin-1',sep=',', 
                  names=["time", "abstract", "id"]))
    #print(i)
    #j+=1


print(len(file))
#print(file[0].head())


import pandas as pd
import numpy as np
import re
import nltk

processed_files=[]
for i in file:
    processed_files.append([preprocessKeywords(text) for text in i['abstract']])


#print(processed_files[0])


def counter(times,i,k):
    cnt = collections.Counter()
    documents=times[i:k+1]
    for i, doc in enumerate(times):
        
        for j in doc:
            
            for word in j.lower().split():
                cnt[word] += 1
    return cnt


def counter_lesswindow(times):# this is for intrvals that not combined with others which is when it less than w-1
    cnt = collections.Counter()
    #documents=times[i:k]
    
    for j in times:
        for word in j.lower().split():
            cnt[word] += 1
    return cnt



k=0
cnt=[]
 
count=0
for i in range(len(processed_files)):
    if(i<w-1):
        #print(times[i], i)
        cnt.append(counter_lesswindow(processed_files[i]))
         
         
        k+=1
    elif(i==w-1):
        #print(times[0:w],s[0:w])
        cnt.append(counter(processed_files[0:w],0,w))
        k+=1
        count+=1
    elif(i>w-1):
        k+=1
        #print(times[count:w+1], s[count:w+1])
        cnt.append(counter(processed_files[count:w+1],count,w+1))
        w+=1
        count+=1



def prob(word,cnt,voc_freq):
    return (cnt[word])/voc_freq

import math
def prob_log(word,count,voc_freq):
    p=prob(word,count,voc_freq)
    return (p*math.log2(p))



entr=[]
for i in range (len(cnt)):
    e=[]
    s=0
    for j in cnt[i]:
        ent=prob_log(j,cnt[i],sum(cnt[i].values()))
        e.append([i,-ent])
        s+=ent
    entr.append(-s)  
     
                






dis = open('entropy_words_sliding_window_2.txt', 'w')
for item in entr:
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


for i in entr:
    y_data.append(i)
    x_data.append(j)
    j+=1

plt.figure(figsize=(20,10))


plt.plot(x_data,y_data)
plt.savefig('words_entropy_window_2.jpg')

