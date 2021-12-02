import sys
import re
import os
import nltk
from nltk.corpus import stopwords
nltk.download('universal_tagset')
import argparse


#parser = argparse.ArgumentParser()

#parser.add_argument("--word",help="find word with highest frequency", required=True)
#parser.add_argument("--number",help="number of highest freq words", required=True)


#args = parser.parse_args()

#word_query = args.word
#n=int(args.number)

def Sort(sub_li):
    l = len(sub_li)
    for i in range(0, l):
        for j in range(0, l-i-1):
            if (sub_li[j][1] > sub_li[j + 1][1]):
                tempo = sub_li[j]
                sub_li[j]= sub_li[j + 1]
                sub_li[j + 1]= tempo
    return sub_li


def get_pos(string):
    string = nltk.word_tokenize(string)
    pos_string = nltk.pos_tag(string,tagset='universal')
    return pos_string




def highest_frequency(word_query):
    #data = dict()
    #data2=dict()
    key={}
    #jj=0
    #data_list=[]
    #max_=0
    word_list=[]

    with open('/zfs/dicelab/news_frequency.txt') as raw_data: 

        for item in raw_data: 
            if ':' in item: 
                key[0]=item.split(',')[0].split('\'')[1]
                key[1]=item.split(',')[1].split('\'')[1]
                value= item.split(':')[1] 
                #data[(key[0],key[1])]=value
                #data_list.append([(key[0],key[1]),value])
                try:
                    if(key[0] ==word_query or key[1]==word_query):
                        word_list.append([(key[0],key[1]),int(value)])
                    #if(int(value)>max_):
                     #   max_=int(value)
                      #  word=(key[0],key[1])
                except:
                    pass
            
    return word_list



def POS_Sort(word_query):
    stops=stopwords.words('english')
    stops.append('is')
    stops.append('was')
    stops.append('are')
    stops.append('were')

    s=highest_frequency(word_query)
    #ss=Sort(s)
    type_=['NNS','NN','JJ','VBD','ADJ','VBN','VERB','NOUN']
    pos_list=[]
    for i in s:
        try:
            k0=get_pos(i[0][0])[0][1]
            k1=get_pos(i[0][1])[0][1]
            if(k0 in type_ and k1 in type_):
                if(i[0][0] not in stops and i[0][1] not in stops):
                    pos_list.append(i)
        except:
            pass

    d=Sort(pos_list)
    if(len(d)>=20):
        return (d[len(d)-20:])
    else:
        return d
         

#word_list_freq=POS_Sort()



def find_words(word_query,n):
    word_list_freq=POS_Sort(word_query)
    k=(word_list_freq[len(word_list_freq)-n:])


    stops=stopwords.words('english')
    stops.append('is')
    stops.append('was')

    words=[]
    for i in k:
        if(i[0][0]!=word_query):
            words.append(i[0][0])
        else:
            words.append(i[0][1])


    print(words)
    return words


find_words('obama',50)
