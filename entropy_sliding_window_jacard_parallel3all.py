import time

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

'''
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



#a=[]
#for  i in processed_files:
 #   for j in i:
  #      a.append(j)

#df=pd.DataFrame(a) 

#df.to_csv("/zfs/dicelab/farah/query_exp/query-construction2/procc_files.csv",index=False)
#file1=pd.read_csv('procc_files.csv', names=['a'])

#print(file1.head())


import csv

with open("/zfs/dicelab/farah/query_exp/query-construction2/procc_files.csv", "w",encoding="utf-8",newline='') as f:
    writer = csv.writer(f)
    writer.writerows(processed_files)

'''


import csv
processed_files=[]
with open("/zfs/dicelab/farah/query_exp/query-construction2/procc_files.csv",'r',encoding='utf-8') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        processed_files.append(row)

print(len(processed_files))
print(len(processed_files[0]))






#print(processed_files[0])

w=5

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


print("Window Done")

import multiprocessing 

# Step 1: Init multiprocessing.Pool()
#pool = mp.Pool(mp.cpu_count())
#print(pool)

#print("Number of processors: ", mp.cpu_count())



import math

def prob_new(word,voc_freq):
    return (word/voc_freq)

import math
def prob_log_new(word,voc_freq):
    p=prob_new(word,voc_freq)
    return (p*math.log2(p))

def get_jaccard_sim(mm,i,j): 
    
    str1=mm[i]
    str2=mm[j]
    a = set(str1.split()) 
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))



import numpy as np
from itertools import combinations

from multiprocessing import Process



def similarities_vectorized2(vector_data,mm,y,q2):
    sim=[]
    print ('func2: starting')
    combs = np.stack(combinations(range(vector_data.shape[0]),2))
    for i in combs:
        sim.append(round(get_jaccard_sim(mm,i[0],i[1]),2))
   # print("done sim fun2")
    cnt = collections.Counter()

    for j in sim:

        cnt[j] += 1

        ent=[]
    #print("done counting")

    s=sum(cnt.values())
    for i in cnt:
        if(i==0):
            continue
        ent.append(prob_log_new(i,s))
    ss=-sum(ent)
    #print("done entropy func 2")
    #dis2.write("%s %i\n" % (ss,y))
    print(ss)
    q2.put([ss,y])
    combs = None
    del combs
    vector_data=None
    del vector_data
    mm=None 
    del mm
    return ss





def similarities_vectorized(vector_data,mm,y,q):
    sim=[]
    print('func1: startting')
    combs = np.stack(combinations(range(vector_data.shape[0]),2))
    for i in combs:
        sim.append(round(get_jaccard_sim(mm,i[0],i[1]),2))
    #print("done sim")
    cnt = collections.Counter()
   
    for j in sim:
        
        cnt[j] += 1
        
        ent=[]
    #print("done counting func1")
    
    s=sum(cnt.values())
    for i in cnt:
        if(i==0):
            continue
        ent.append(prob_log_new(cnt[i],s))
    ss=-sum(ent)
   # print("done entropy fun 1")
    #dis.write("%s\n" % ss+ "  " +y)
   # dis.write("%s %i\n" % (ss,y))
    print(ss, y)
    q.put([ss,y])
    combs = None
    del combs
    vector_data=None
    del vector_data
    mm=None
    del mm

    return ss 

def print_queue(q,filename):
    """
    function to print queue elements
    """
    global s
    print("Queue elements:")
    dis=open(filename,'w')
    while not q.empty():
        for i in q.get():
            dis.write('%s ' % i)
        dis.write('\n')
    dis.close()
    print("Queue is now empty!")






q = multiprocessing.Queue()
#q2 = multiprocessing.Queue()


#processes = [Process(target=rand_num, args=()) for x in range(4)]

'''
simlar=[]
kk=0
for i in range(int(len(wind)/50)):
    #print(i)
   # if(kk==40):
    #    p1.join()
    p1 = Process(target=similarities_vectorized,args=(np.asarray(wind[i]),wind[i],i,q))
    
    p1.start()
    #print(p1)
   # p1.join()
    #similarity = similarities_vectorized(np.asarray(wind[i]),wind[i])
    #simlar.append(similarity)
    simlar.append(p1)
    kk+=1
    del wind[i]


#p1.join()


for i in range(int(len(wind)/50),int(len(wind)/50)+30):
    p2 = Process(target=similarities_vectorized,args=(np.asarray(wind[i]),wind[i],i,q))
    p2.start()
    simlar.append(p1)
    del wind[i]


p2.join()

'''





a=[]
for i in range(52):
    a.append(i*30)


# time.sleep(5)

print(a[6])

print(a[10])
def pr(j):
    print(j)

#52

for i in range (37,39):
    if(i==51):
        break
    for j in range(a[i],a[i+1]):
       # print(j)
        ar=np.asarray(wind[j])
        p1 = Process(target=similarities_vectorized,args=(ar,wind[j],j,q))
       # #p1=Process(target=pr,args=(j))
        p1.start()
        del wind[j]
        del ar
    p1.join()
    #print("***************************************************************************")




    ##time.sleep(120)




'''
simlar2=[]
for i in range(int(len(wind)/2),len(wind)):
    #print(i)
    if(i==40):
        break
    p2 = Process(target=similarities_vectorized2,args=(np.asarray(wind[i]),wind[i],i,q2))
    p2.start()
   # p2.join()
    #print(p2)
    #similarity = similarities_vectorized(np.asarray(wind[i]),wind[i])
    #simlar.append(similarity)
    simlar.append(p2)

#p1.join()

p2.join()


'''

f1='/zfs/dicelab/farah/query_exp/query-construction2/ent_sliding_jacard_parallel1_new.txt'

f2='/zfs/dicelab/farah/query_exp/query-construction2/ent_sliding_jacard_parallel2.txt'
img='/zfs/dicelab/farah/query_exp/query-construction2/ent_sliding_jacard2.jpg'


p3 = multiprocessing.Process(target=print_queue, args=(q,f1))
p3.start()
p3.join()

#p4 = multiprocessing.Process(target=print_queue, args=(q2,f2))
#p4.start()
#p4.join()




#import plot_parallel as p

#p.fun(f1,f2,img)


