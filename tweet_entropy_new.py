
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
import argparse
from os import listdir
import argparse
from os import listdir
import pandas as pd
import numpy as np
import re
import nltk
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

#parser = argparse.ArgumentParser()
#args = parser.parse_args()
#input_files = os.listdir(args.input_dir)




import csv, subprocess
import time
import os
import sys
import argparse
from os import listdir

STOPWORDS = dec_text.getStopwords()


import pandas as pd
#corpus = pd.read_csv('clean_police_all_tweet.txt',encoding='latin-1',sep=',',
 #                 names=["abstract"])


#corpus = pd.read_csv('t_clean_new.txt',encoding='latin-1',sep=',',
 #                 names=["abstract"])


#t_clean_new.txt

#print(len(corpus))



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


data=pd.read_csv('/zfs/dicelab/farah/query_exp/query-construction2/Baltimore/police_tweet.csv',encoding='latin-1',sep=',',names=["time", "abstract", "id"])

#f1=pd.read_csv('/zfs/dicelab/farah/query_exp/query-construction2/sample/file_1.csv',encoding='latin-1',sep=',',names=["time", "abstract", "id"])

#f1 = pd.read_csv('/zfs/dicelab/farah/query_exp/query-construction2/sample/file_1.csv',encoding='latin-1',sep=',',
 #                 names=["time","abstract","id"])


#f1=pd.read_csv("/zfs/dicelab/farah/query_exp/query-construction2/sample/file_1.csv",encoding='latin-1',sep=',',names=["time", "abstract", "id"])
#f2=pd.read_csv('/zfs/dicelab/farah/query_exp/query-construction2/sample/file_11.csv',encoding='latin-1',sep=',',names=["time", "abstract", "id"])

#f3=pd.read_csv('/zfs/dicelab/farah/query_exp/query-construction2/sample/file_13.csv',encoding='latin-1',sep=',',names=["time", "abstract", "id"])

cor=[]
for i in data['abstract']:
    cor.append(i)
#cor=[]
#for i in f1['abstract']:
 #  cor.append(i)

#for i in f2['abstract']:
 #  cor.append(i)

#for i in f3['abstract']:
 #  cor.append(i)

#cor=([preprocessKeywords(text) for text in corpus['abstract']])

cor=([preprocessKeywords(text) for text in cor])

import collections
cnt_corpus = collections.Counter()

for i, doc in enumerate(cor):
    cnt_corpus[doc] += 1


#print(cnt_corpus)

#l=len(corpus)



f = open('dict_all_poilice_new.txt',"w")
f.write( str(cnt_corpus) )
f.close()



import ast


#with open('dict_all_poilice.txt', 'r') as f:
 #   s = f.read()
  #  dic = ast.literal_eval(s)

#print(dic)
#cnt_corpus=dic




def prob(word,blob,bloblist,cnt2):
    #if cnt2[word]!=0:
    print(cnt2[word])
    return (cnt2[word]/len(bloblist))
    #else: return 0

def entropy_fun(word,blob,bloblist,cnt2):
    p=prob(word,blob,bloblist,cnt2)
    #if p!=0:
    return -((np.log2(p))*p)
    #else: return 0
 
def compute_entropy(input_files=[],input_dir=''):
        """
        compute entropy for a query result related to topic i
        :return average entropy per tweet of the input_files
        """
        ans=[]
        

        split_numbers=[]
        for i in input_files:
            split_numbers.append(int(i.split('_')[1].split('.')[0]))
        print(split_numbers)
        sorted_num=sorted(split_numbers)
        files=[]
        for i in sorted_num:
            files.append("file"+"_"+str(i)+".csv")
        
        for interval, f in enumerate(files):
            #entropy = 0.0
                ss=0
                ent=[]
                #tweet_cnt = 0
                print("Processing data from file " + f)
                inputf0 = os.path.join(input_dir,f)
                file=pd.read_csv(inputf0,encoding='latin-1',sep=',', names=["time", "abstract", "id"])                
                #print(file.head())
                processed_file=[preprocessKeywords(text) for text in file['abstract']]
                
                for i, doc in enumerate(processed_file):
                    ent.append(entropy_fun(doc,doc,cor,cnt_corpus))
                    print(doc) 
                for i in ent:
                    ss+=i
                print(int(f.split('_')[1].split('.')[0]),ss/len(file))
                ans.append([int(f.split('_')[1].split('.')[0]),ss/len(file)])
        return ans



if __name__ == "__main__":
        # get input files
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_dir",help="input directory of query result files to count hashtags", required=True)
        #parser.add_argument("--output_file",help="output file for entropy measure in query results", required=True)
        #parser.add_argument("--wordcount_input_file",help="file to perform word count", required=True)
        args = parser.parse_args()
        input_files = os.listdir(args.input_dir)
        #output_f = open(args.output_file,'w')
        #get word count map and save it in freqcount
        #wordcount_file = args.wordcount_input_file
        '''
	freqcount = dec_text.getWordCount(file=wordcount_file) # this will return the frequency of words in all the vocab in all files
        f = open("dict_pal.txt","w")
        f.write( str(freqcount) )
        f.close()
        '''
        input_topici = [x for x in input_files]
        entropy = compute_entropy(input_files=input_topici,input_dir=args.input_dir)
        print(entropy)
        dis = open('tweet_entropy_key_value.txt', 'w')
        for item in entropy:
            dis.write("%s\n" % item)
        dis.close()
        '''
        #measure entropy of each topic's query results
        for i in range(args.num_topics):
                #input_topici = [x for x in input_files if "topic"+str(i) in x]
                input_topici = [x for x in input_files]
                #output_topici = args.output_file+"_topic"+str(i)
                entropy = compute_entropy(input_files=input_topici,input_dir=args.input_dir,freqcount=freqcount)
                #output_f.write("entropy of query results from topic "+str(i) + " is "+"\n"+str(entropy)+"\n")
        output_f.close()


	'''

