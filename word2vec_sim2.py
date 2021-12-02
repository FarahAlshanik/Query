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
import os
import numpy as np
#import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()
def tokenize(tweet):
  
        #tweet = unicode(tweet.decode('utf-8').lower())
        tokens = tweet.split(" ")
        
        return tokens

   
def postprocess(data):
   
    data['tokens'] = data.map(tokenize)  ## progress_map is a variant of the map function plus a progress bar. Handy to monitor DataFrame creations.
    #data = data[data.tokens != 'NC']
    #data.reset_index(inplace=True)
    #data.drop('index', inplace=True, axis=1)
    return data





import csv, subprocess
import time
import os
import sys
import argparse
from os import listdir

STOPWORDS = dec_text.getStopwords()


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


data=pd.read_csv('/zfs/dicelab/farah/query_exp/query-construction2/Baltimore/sorted.csv',encoding='latin-1',sep=',',names=["time", "abstract", "id"])

cor=[]
for i in data['abstract']:
    cor.append(i)


#data = postprocess(cor) 

def build_corpus(data):
    "Creates a list of lists containing words from each sentence"
    corpus = []
    for sentence in data:
        word_list = sentence.lower().split()
        corpus.append(word_list)    
           
    return corpus

corpus = build_corpus(cor)


model_all=word2vec.Word2Vec(corpus, size=100, window=5, min_count=5, workers=4)
model_all.save("/zfs/dicelab/farah/dec_results/word2vec_alltweet.model")
