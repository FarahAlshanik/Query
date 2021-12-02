"""
Text cleaning procedures implementation of 
"https://people.cs.clemson.edu/~isafro/papers/dynamic-centralities.pdf"

Grace Glenn (mgglenn@g.clemson.edu)
November 2017
"""
import csv
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import numpy as np

def getEntropy(text=[],freqcount={}):
	"""
	compute entropy of given text
    :param text a list of list of words text[0] ['boston','riot']
    :param freqcount map for the vocabulary
	:return entropy of the text
	"""
	entropy = 0.0
	voc_len = sum(freqcount.values())
#	print(voc_len)
	for seq in text:
		for word in seq:
			#print(word)
			probability = freqcount[word] / float(1.0 * voc_len)
			self_information = np.log2(1.0/probability)
			if self_information < 0:
				print("word " + word + " has self information "+str(self_information) + " has freq count "+str(freqcount[word]))
			entropy += (probability * self_information)
#	print(entropy)
	return entropy


def getEntropyTweet(le=0,text=[],freqcount={}):#ME
        """
        compute entropy of given text
    :param text a list of list of words text[0] ['boston','riot']
    :param freqcount map for the vocabulary
        :return entropy of the text
        """
        entropy = 0.0
        voc_len = le
#       print(voc_len)
        for seq in text:
            #print(seq)
            line=' '.join(seq)
            print(line)
                #for word in seq:
                        #print(word)
           # print(freqcount[line])     
           # probability = freqcount[seq] / float(1.0 * voc_len)
           # self_information = np.log2(1.0/probability)
                 #if self_information < 0:
                 #print("word " + word + " has self information "+str(self_information) + " has freq count "+str(freqcount[word]))
            #entropy += (probability * self_information)
     #   print(entropy)
      #  return entropy




def getTweetCount(file=''):#Me
        """
        get tweet count from preprocessed file
        :param file: preprocessed input file
        :return word count dictionary
        """
        wcfile = open(file,'r')
        import pandas as pd 
        data=pd.read_csv(file,encoding='latin-1',sep=',',names=["ab"])
        freqcount = {}
        #for line in wcfile:
        for line in data['ab']:
           # print(line)
            #words = line.strip()
                #for word in words:
            freqcount[line] = freqcount.get(line,0)+1
        print(freqcount['rt ijessewilliams want condemn black folk violent property never condemn police killing actual people']) 
        #print(freqcount)
        return freqcount




def getWordCount(file=''):
	"""
	get word count from preprocessed file
	:param file: preprocessed input file
	:return word count dictionary
	"""
	wcfile = open(file,'r')
	freqcount = {}
	for line in wcfile:
		words = line.strip().split()
		for word in words:
			freqcount[word] = freqcount.get(word,0)+1
	return freqcount

def getStopwords(file='/zfs/dicelab/farah/query_exp/query-construction2/stopwords.txt'):
	"""
	Load in stop words.
	"""
	stopwordfile = open(file, 'r')
	stopwordlist = []
	for line in stopwordfile:
		for word in line.split():
			stopwordlist.append(word)
	return stopwordlist


def removeStopwords(text=[], stopwords=[]):
	"""
	Remove stopwords from a list of text.
	"""
	text = [x for x in text if x.lower() not in stopwords]
	text = [x for x in text if len(x) > 1]
#	text = " ".join(text) #only this extra
#	print(text)
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
	#	text.append( " ".join(stemmed))
#	print(stemmed)
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

def getHashTag(text=[]):
	"""
	Get HashTags and count frequency
	"""
	hashtags = []
	for word in text:
		if word.startswith("#"):
			hashtags.append(word)
	return hashtags

def preprocessKeywords(text=[], stopwords=[]):
	"""
	Process all tokens.
	:param text: word tokens representing a document (ie tweet).
	:param stopwords: stopwords to filter on.
	:returns text: cleaned text
	"""
	text = removeHashTag(text)
	text = stemText(text)
	text = removeStopwords(text, stopwords=stopwords)  # remove commit here
	return text


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

def get_text_from_news_file(file=None, stopwords=[]):
	"""
	Takes all tweets from a given file and returns data, list of all keyword lists.
	:param file: file to process
	:returns data: list of lists

	eg
	# data[0] = ['boston', 'marathon', 'broadcast', 'live']
	"""
	data = []
	with open(file, 'r') as csvfile:
		for line in csvfile:
			tweetData = line.strip()
			tweet = getWords(tweetData)
			tweetKeywords = preprocessKeywords(tweet, stopwords=stopwords)

			# make sure we have at least one pair (edge)
			if len(tweetKeywords) > 1:
				data.append(tweetKeywords)
	return data



def get_text_from_file(file=None, stopwords=[]):
	"""
	Takes all tweets from a given file and returns data, list of all keyword lists.
	:param file: file to process
	:returns data: list of lists

	eg
	# data[0] = ['boston', 'marathon', 'broadcast', 'live']
	"""
	data = []
	with open(file, 'r') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			tweetData = line[1]
			tweet = getWords(tweetData)
			tweetKeywords = preprocessKeywords(tweet, stopwords=stopwords)

			# make sure we have at least one pair (edge)
			#if len(tweetKeywords) > 1:
			data.append(tweetKeywords)
	return data


def write_dec_values(outfile='', dec_vals=None, rank=False):
	"""
	Writes dynamic ecentrality values to a given file.
	:param outfile: file to write to
	:param dec_vals: dictionary of word->dec pairs
	:returns: number of keywords read
	"""
	items = dec_vals.items()

	if rank:
		# arrange keywords from highest to lowest DEC value
		items = sorted(items, key=lambda x: x[1], reverse=True)
		samp = items[:5]
		print("\tTop five keywords: ")
		for pair in samp:
			print("\t\t" + str(pair))

	keywords = 0
	ecentral = open(outfile,'w')
	for word, dec in items:
		ecentral.write(word)
		ecentral.write(" ")
		ecentral.write(str(dec))
		ecentral.write('\n')
		keywords += 1

	return keywords
