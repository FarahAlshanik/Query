
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

data=[]

r=0
    
for o in range(len(processed_files)):
    for u in processed_files[r]:
        data.append(u)
    r+=1
    #print(mm)
    

def build_corpus(data):
    "Creates a list of lists containing words from each sentence"
    corpus = []
    for sentence in data:
        word_list = sentence.lower().split()
        corpus.append(word_list)    
           
    return corpus

tokens=build_corpus(data)

 
from gensim.models import Word2Vec
model_all=Word2Vec(tokens, size=100, window=5, min_count=0, workers=4)


print("Done Combine corpus for word2vec")

ee_all=[]
for i, word in enumerate(model_all.wv.vocab):

    ee_all.append([word,model_all[word]])

#ee_file1_2
emb_all={}
for i in ee_all:
    #print(i[0])
    emb_all[i[0]]=i[1]


#https://www.codingame.com/playgrounds/6233/what-is-idf-and-how-is-it-calculated
#this code for compute the idf only

#lowest idf is noise like the is a 
#Loowes idf like domain based 

def extract_features( document ):
   terms = tuple(document.lower().split())
   features = set()
   for i in range(len(terms)):
      for n in range(1,2):
          #if i+n <= len(terms):
            features.add(terms[i:i+n])
   return features


def calculate_idf( documents ):
   N = len(documents)
   from collections import Counter
   tD = Counter()
   for d in documents:
      features = extract_features(d)
      for f in features:
          tD[" ".join(f)] += 1
   IDF = []
   import math
   for (term,term_frequency) in tD.items():
       term_IDF = math.log2(float(N) / term_frequency)
       IDF.append(( term_IDF, term ))
       #print(term,term_frequency)
   IDF.sort(reverse=True)
   return IDF

    

#######################For tweet Entropy
def idf(data):
    idf_file_data=[]
    
    for (IDF, term) in calculate_idf(data):
        idf_file_data.append([IDF, term])
    return idf_file_data

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



idf_data=[] #for each window
for i in wind:
    idf_data.append(idf(i))
    

def idf_dic(t):
    idf_dict={}
    for i in t:
        idf_dict[i[1]]=i[0]
    return idf_dict
    

idf_data_dict=[]
for i in range(len(idf_data)):
    idf_data_dict.append(idf_dic(idf_data[i]))

  #check this I think the word centroid similarity does not take avg not divide by total frq
def get_centroid_idf(corpus, emb, idf, D): # to find the word centroid similarity   
    # corpus here is document
    centroid = np.zeros((1, 100))
    div = 0
    #D is number of dimension
    # Computing Terms' Frequency
    
    tf = defaultdict(int)
    tokens = corpus.split()
    for word in tokens:
        #print(word)
        if word in emb:
            tf[word] += 1
            #print(tf[word])

    # Computing the centroid
   
 

        for word in tf:
            if word in idf:
                p = tf[word] * idf[word]
                centroid = np.add(centroid, emb[word]*p)
                div += p
    if div != 0:
        centroid = np.divide(centroid, div)
    return centroid

def get_avg(data,k):
    #s=[]
    #for i in data:
     #   for j in i:
      #      s.append(j)
    avg_file1=[]
    for i in data:
        avg_file1.append(get_centroid_idf(i,emb_all,idf_data_dict[k],100))
        
    final_cent_array_file1=[] # now I got the word centroid for each doc 
    for i in avg_file1:
        final_cent_array_file1.append( np.array(i, dtype=np.float32).reshape(( 100)))
        
    return final_cent_array_file1
    
    
from collections import defaultdict

get_average=[]#centroied for each tweet in the window
for j,i in enumerate(wind):
    #print(i)
    get_average.append(get_avg(i,j)) 
    
 # we need to find average for all

from scipy import spatial







def sim_embedding(w):
    similarity=[]
    for i in range(len(w)):
        for j in range(i+1,len(w)):
            sim=1-spatial.distance.cosine(w[i],w[j])
            similarity.append(sim)
            
            
    return similarity
        

simlar=[]
for i in range(len(get_average)):
    simlar.append(sim_embedding(get_average[i]))






import math

def prob_new(word,voc_freq):
    return (word/voc_freq)

import math
def prob_log_new(word,voc_freq):
    p=prob_new(word,voc_freq)
    return (p*math.log2(p))

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




dis = open('/zfs/dicelab/farah/query_exp/query-construction2/entropy_tweet_sliding_window_cosine.txt', 'w')
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
plt.savefig('/zfs/dicelab/farah/query_exp/query-construction2/tweet_entropy_window_cosine.jpg')

