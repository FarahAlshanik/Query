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
    print(split_numbers)
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
from nltk.corpus import stopwords
from nltk.stem.porter import *
stemmer = PorterStemmer()
import os

processed_files=[]
for i in file:
    processed_files.append([preprocessKeywords(text) for text in i['abstract']])


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
    idf_file.append("idf_file_"+str(i))

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

all_corpuss=[] 
for i in corpus_file:
    all_corpuss+=i
from gensim.models import Word2Vec
model_all=Word2Vec(all_corpuss, size=100, window=5, min_count=0, workers=4)

ee_all=[]
for i, word in enumerate(model_all.wv.vocab):
    
    ee_all.append([word,model_all[word]])

#ee_file1_2
emb_all={}
for i in ee_all:
    #print(i[0])
    emb_all[i[0]]=i[1]


#all journal
#k=0 
dict_idf = {}
for i in dict:
    print(i)
    k=int(i.split("_")[2])
    print(k)
    dict_idf[idf_file[k]]={}
    for j in dict[i]:
     
        dict_idf[idf_file[k]][j[1]]=j[0]
    #k+=1



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
dict_avg_files_all = {}

 
for i in processed_files:
    if(k==len(processed_files)):
        break
    dict_avg_files_all[idf_file[k]]=[]
    for j in i:
        dict_avg_files_all[idf_file[k]].append(get_centroid_idf(j,emb_all,dict_idf[idf_file[k]],100))
    k+=1

#all journal
#k=0 
final_cent_array_file_all = {}
for i in dict_avg_files_all:
    k=int(i.split("_")[2])
    print(k)
    final_cent_array_file_all[idf_file[k]]=[]
    for j in dict_avg_files_all[i]:
        final_cent_array_file_all[idf_file[k]].append(np.array(j, dtype=np.float32).reshape(( 100)))
    #k+=1




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

lenn=len(final_cent_array_file_all)
from scipy import spatial

centroid_dic={}
for q in range(len(final_cent_array_file_all)):
    print(q)
    for i in final_cent_array_file_all[idf_file[q]]:
        a=sum(final_cent_array_file_all[idf_file[q]])
        b=a/len(final_cent_array_file_all[idf_file[q]])
        centroid_dic[q]=b
        


sim_dic=[]

for i in range(0,lenn):
    k=i+1
   #if(k==lenn-1):
    #        break
    for j in range(i+1,lenn):
        if(k==lenn):
            break
        sim=1-spatial.distance.cosine(centroid_dic[i],centroid_dic[j])
        sim_dic.append([i,j,sim])
        k+=1
        




    
    
dis = open('similarity_centroid.txt', 'w')
for item in sim_dic:
  dis.write("%s\n" % item)
dis.close()


