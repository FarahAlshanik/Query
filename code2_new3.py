import collections
import numpy as np
from itertools import combinations

import numpy as np
import argparse
from os import listdir
from os.path import isfile, join
import sys
import re
import os

parser = argparse.ArgumentParser()

parser.add_argument("--input",help="sliding window for entropy", required=True)

args = parser.parse_args()

w=(args.input)


def prob_new(word,voc_freq):
    return (word/voc_freq)

import math
def prob_log_new(word,voc_freq):
    p=prob_new(word,voc_freq)
    return (p*math.log2(p))



import numpy as np
from itertools import combinations


import numpy as np
from itertools import combinations

def get_jaccard_sim(mm,i,j): 
    
    str1=mm[i]
    str2=mm[j]
    #print(str1)
    #print(str2)
    #print(i)
    #print(j)
    a = set(str1.split()) 
    b = set(str2.split())
    c = a.intersection(b)
    try:
        sim=round((len(c)) / (len(a) + len(b) - len(c)),2)
    except ZeroDivisionError:
        sim=0

    if(sim>=0.25):
        return sim,str2    
    else:
        return '',''




def similarities_vectorized(mm):
    sim=[]
    cluster=[]
    simi=0
    non=0
 
    #combs = np.stack(combinations(range(vector_data.shape[0]),2))
    mer=[]
    for i in range (len(mm)):
        for j in range(len(mm)):
            mer.append([i,j])
   # print(mer)        
    combs=mer
    
    #print(combs)
    #print(combs)
    sim_combined = {}
    sim_combined_tweets = {}
    n=0
    r=0
    sim_tweets=[]
    non_sim_twetts=[]
    for i in mer:
        if(n==len(mm)): 
            #sim_combined[r]=[] 
           # print("**********************************************")
            n=0
           # print(sim)
            
            
            
            #sim_combined[r]=sim
            sim_combined[r]=[i for i in sim if i]
            sim_combined_tweets[r]=[i for i in sim_tweets if i]
            sim=[]
            sim_tweets=[]
            
            r+=1
        sss,rrr=(get_jaccard_sim(mm,i[0],i[1]))
        sim.append(sss)
        sim_tweets.append(rrr)
        n+=1
    
    return sim_combined,sim_combined_tweets #to eturn similarity for each tweet imagine the tweet is a cluster and then find the similar tweet for this cluster
        
    


def fun(str):
	with open(str+".txt"  , 'r',newline='') as f:
		x = f.read().splitlines() 
	s=[]
	for i in (x):
        	s.append(i)


	print(s[0])
	print(s[1])


	similarity,sim_tweets = similarities_vectorized(s)
	#print(sim_tweets)
	print(len(sim_tweets))
	print(sim_tweets[1])

	k=[]
	for i in sim_tweets:
        	k.append(sim_tweets[i])       

	print(len(k))

#	print(similarity)	
	Output = {} 

	for lis in k: 
    		Output.setdefault(tuple(lis), list()).append(1) 
	for a, b in Output.items(): 
    		Output[a] = sum(b) 

	print('out',len(Output))

	values=[]
	keys=[]
	for i in Output.keys():
    		keys.append(i)

	
	print('len keys',len(keys))

	for i in Output.values():
		values.append(i)
	print('len values',len(values))

	t_above_one=[]
	t_less=[]

	tweet=[]
	for i in range(len(values)):
    		if(values[i]>1):
        		t_above_one.append([values[i],(keys[i]),len(keys[i])])# it has number of classes that have the same tweets, tweets , number of tweet in each class
    		else:
        		t_less.append([values[i],(keys[i]),len(keys[i])]) 
        		tweet.append(keys[i])   
  

	print("above 0ne", len(t_above_one))
	print('less one', len(t_less))

	classes=t_above_one+t_less
	
	print("class",len(classes))

	data_sets=[]
	for i in classes:
    		data_sets.append(list(i[1]))

	from itertools import combinations
	sets=[]
	for i in classes:
    		for j in i[1]:
        		sets.append(j)


	inter=set([x for x in sets if sets.count(x) > 1])


	p=0
	ent=[]
	for i in data_sets:
		res=list(filter(lambda o: o not in inter, i) )
    
		bb=len(res)
		#print(bb)
		if(bb!=0):
			p+=bb/ len(set(s))
			ent.append(prob_log_new(bb,len(set(s))))
    
	if(len(inter)!=0):    
		p=p+(len(inter)/len(set(s)))
		ent.append(prob_log_new(len(inter),len(set(s))))


	print("sum of probablity = ",p)

	output_filename = 'Results_%s.txt'% (w);

	entr=-sum(ent)

	dis=open(output_filename,'w')
	dis.write("%s %s" % (entr,w))
	dis.write('\n')
	dis.close()



fun(w)

