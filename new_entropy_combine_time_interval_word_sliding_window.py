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
import collections
#parser = argparse.ArgumentParser()
#args = parser.parse_args()
#input_files = os.listdir(args.input_dir)
#input_files =  get_files(file_folder="/zfs/dicelab/farah/break_police_15M")

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir",help="input directory of query result files to count hashtags", required=True)
args = parser.parse_args()
input_files = os.listdir(args.input_dir)

#input_files =  get_files(file_folder="/zfs/dicelab/farah/query_exp/query-construction2/sample")

#input_dirc='/zfs/dicelab/farah/query_exp/query-construction2/sample/'
input_dirc='/zfs/dicelab/farah/break_police_15M/'

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
    full_path.append(input_dirc+f)
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

def prob(word,cnt,voc_freq):
    return (cnt[word])/voc_freq 

import math
def prob_log(word,count,voc_freq):
    p=prob(word,count,voc_freq)
    return (p*math.log2(p))
 

entropy=[]
for i, _ in enumerate(processed_files):
    entropy.append("entropy"+str(i))


#cnt_times = collections.Counter()
dict = {} # to stor the counter for time interval
w=5
r=0
k=0
len_files=[]# to have the number of documents in each 2 time interval
s=0
#count_times=[]
for i in range(len(processed_files)):
    dict[entropy[k]]=collections.Counter()
    print(k)
    if(k==0):
        s+=len(processed_files[i])
    for doc in (processed_files[i]):
        for word in doc.lower().split():
             dict[entropy[k]][word]+=1
    if(k!=0 and r%w!=0):
        dict[entropy[k]]=dict[entropy[k]]+dict[entropy[k-1]]
        s+=len(processed_files[i])
        len_files.append(s)
    else:
        dict[entropy[k]]=dict[entropy[k]]

    k+=1
    r+=1





freq=[]

for i in  range(len(entropy)):
     freq.append(sum(dict[entropy[i]].values()))






m=0
entropy_val=0
k=0
ent=[]
for k in range (len(processed_files)):
    m=0
    entropy_val=0
    print(k)
    for i in dict[entropy[k]]:# to take the uniqe words only 
       # print(i)
        entropy_val=(prob_log(i,dict[entropy[k]],freq[k]))
       # print(entropy_val)
        m+=entropy_val
    ent.append([k,-m])








dis = open('new_entropy_combine_interval_words_sliding_window.txt', 'w')
for item in ent:
  dis.write("%s\n" % item)
dis.close()

