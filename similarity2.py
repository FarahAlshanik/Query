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





def get_files(file_folder='', file_format='file_%d.csv'):
        """
        Return all interval files in a given folder.
        Usage:
                #files = get_files('/zfs/dicelab/farah/query_exp/query-construction2/Baltimore/interval/int/', 'file_%d.csv')
                files=get_files('/zfs/dicelab/farah/query_exp/query-construction2/Baltimore/interval/queryoutput/lda_int15/', 'file_%d.csv')
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


import argparse
from os import listdir
import argparse
from os import listdir

#parser = argparse.ArgumentParser()
#args = parser.parse_args()
#input_files = os.listdir(args.input_dir)
#input_files =  get_files(file_folder="/zfs/dicelab/farah/break_police_15M")
input_files =  get_files(file_folder="/zfs/dicelab/farah/query_exp/query-construction2/sample")

j=0
file=[]
for i in input_files:
    print(i)
    file.append(pd.read_csv(i,encoding='latin-1',sep=',', 
                  names=["time", "abstract", "id"]))
    #print(i)
    #j+=1

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
stemmer = PorterStemmer()
import os


processed_file=[]
for i in file:
    processed_file.append([preprocessKeywords(text) for text in i['abstract']])

#processed_files=[]
#for i in processed_file:
 #   processed_files.append([removeHashTag(text) for text in i])

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

documents = [
   "This article is about the Golden State Warriors",
   "This article is about the Golden Arches",
   "This article is about state machines",
   "This article is about viking warriors"]
 


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


idf_file=[]
for i, _ in enumerate(processed_files):
    idf_file.append("idf_file_"+str(i+1))

#all journal
k=0 
dict = {}
for i in processed_files:
    dict[idf_file[k]]=[] 
    for (IDF, term) in calculate_idf(i):
        dict[idf_file[k]].append ([IDF, term])
    k+=1

from gensim import utils
# Import required libraries
import pandas as pd
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score

def build_corpus(data):
    "Creates a list of lists containing words from each sentence"
    corpus = []
    for sentence in data:
        word_list = sentence.lower().split()
        corpus.append(word_list)    
           
    return corpus


corpus_file = []
for i in processed_files:
    corpus_file.append(build_corpus(i))

model_files=[]
j=1
for i in corpus_file:
    if(j==len(corpus_file)):
        break
    model_f1_f2=i+corpus_file[j]
    print(len(model_f1_f2)) 
    model_files.append( word2vec.Word2Vec(model_f1_f2, size=100, window=5, min_count=5, workers=4))
    j+=1


#all journal
k=0 
dict_idf = {}
for i in dict:
    dict_idf[idf_file[k]]={}
    for j in dict[i]:
     
        dict_idf[idf_file[k]][j[1]]=j[0]
    k+=1


#all journal
k=0 
dict_ee = {}
for i in model_files:
    dict_ee[idf_file[k]]=[] 
    for j, word in enumerate(i.wv.vocab):

        
        dict_ee[idf_file[k]].append([word,i[word]])
    k+=1


#all journal
k=0 
dict_emb_file = {}
for i in dict_ee:
    dict_emb_file[idf_file[k]]={}
    for j in dict_ee[i]:
     
        dict_emb_file[idf_file[k]][j[0]]=j[1]
    k+=1


 #check this I think the word centroid similarity does not take avg not divide by total frq
def get_centroid_idf(corpus, emb, idf, D): # to find the word centroid similarity
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

from collections import defaultdict
k=0 
dict_avg_files = {}

 
for i in processed_files:
    dict_avg_files[idf_file[k]]=[]
    for j in i:
        dict_avg_files[idf_file[k]].append(get_centroid_idf(j,dict_emb_file[idf_file[k]],dict_idf[idf_file[k]],100))
    k+=1


#all journal
k=0 
final_cent_array_file = {}
for i in dict_avg_files:
    final_cent_array_file[idf_file[k]]=[]
    for j in dict_avg_files[i]:
        final_cent_array_file[idf_file[k]].append(np.array(j, dtype=np.float32).reshape(( 100)))
    k+=1


def cos_sim(a,b):
    dot = np.dot(a, b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    cos = dot / (norma * normb)
    return cos


#           file2
#file1      doc1   doc2  avg
#doc1 
#doc2
#doc3
 # for only 2 file
l=0
sim_dic_2={}
k=0
y=0
w=1
r=[]



for q in range(len(final_cent_array_file)-1):
    for i in final_cent_array_file[idf_file[y]]:
        sum_=0
        l=0
        for j in final_cent_array_file[idf_file[w]]:
        #cos_sim(i,j)
        #sim_dic[k].append(cos_sim(i,j))
            sim=cos_sim(i,j)
            if(str(sim)!='nan'):
                sum_+=sim
                l+=1
        if(l!=0):
            #print(l)
            sim_dic_2[k]=sum_/l
        k+=1
        
    ss=0
    for i in sim_dic_2:
        ss+=(sim_dic_2[i])

    ss=ss/len(sim_dic_2)
    r.append([y,w,ss])
    
    y=y+1
    w+=1
    
    
dis = open('similarity_sample.txt', 'w')
for item in r:
  dis.write("%s\n" % item)
dis.close()


