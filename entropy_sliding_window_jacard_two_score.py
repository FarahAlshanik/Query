
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



def counter(t):
    mm=[]
    documents=t
    #print(t)
    r=0
    
    for o in range(len(documents)):
        for u in documents[r]:
            mm.append(u)
        r+=1
    #print(mm)
    return mm




k=0
wind=[]
 
count=0
for i in range(len(processed_files)):
    if(i<w-1):
        #print(times[i], i)
        wind.append(processed_files[i])

         
         
        k+=1
    elif(i==w-1):
        #print(times[0:w],s[0:w])
        wind.append(counter(processed_files[0:w]))
        k+=1
        count+=1
    elif(i>w-1):
        k+=1
        #print(times[count:w+1], s[count:w+1])
        wind.append(counter(processed_files[count:w+1]))
        w+=1
        count+=1




import math

def prob_new(word,voc_freq):
    return (word/voc_freq)

import math
def prob_log_new(word,voc_freq):
    p=prob_new(word,voc_freq)
    return (p*math.log2(p))


def get_jaccard_sim(str1, str2): 
    
    a = set(str1.split()) 
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))




def sim(w):
    similarity=[]
    for i in range(len(w)):
        for j in range(i+1,len(w)):
            similarity.append(get_jaccard_sim(w[i],w[j]))
            #print(i,j)
    return similarity
        
    

simlar=[]
for i in range(len(wind)):
    simlar.append(sim(wind[i]))


def grade(similarity):
    A=0
    B=0
    
    for i in similarity:
        if(i>=0.75):
           A+=1
        else:
           B+=1

         
    cn=[A,B]
    return cn
    

grades=[]
for i in range(len(simlar)):
    grades.append(grade(simlar[i]))

entr=[] 
for i in range (len(grades)):
    ent=[]
    for j in grades[i]:
        if(j==0):
            continue
        ent.append(prob_log_new(j,sum(grades[i])))
    entr.append(-sum(ent))




dis = open('/zfs/dicelab/farah/query_exp/query-construction2/entropy_tweet_sliding_window_jacard_two_score.txt', 'w')
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
plt.savefig('/zfs/dicelab/farah/query_exp/query-construction2/tweet_entropy_window_jacard_two_score.jpg')

