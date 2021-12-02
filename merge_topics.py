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
import os
import pandas as pd
import datetime

def get_files(file_folder=''):
      
                
       
        if len(file_folder):
                if file_folder[-1] != '/':
                        file_folder += '/'

        num_files = len([f for f in listdir(file_folder) if isfile(join(file_folder, f))])
        print(num_files)
        
        print(sorted(listdir(file_folder)))
        files = []
        for i in (sorted(listdir(file_folder))):
                files.append(file_folder+i )
                #print(files[-1])

        return files

import argparse
from os import listdir
import argparse
from os import listdir


parser = argparse.ArgumentParser()

parser.add_argument("--input_dir",help="input directory of query result files to count hashtags", required=True)
parser.add_argument("--n",help="number of topics", required=True)
parser.add_argument("--output_dir", help="directory of merged files",required=True)

args = parser.parse_args()

input_files = os.listdir(args.input_dir)

print(input_files)

#input_files =  get_files(file_folder="C:\\Users\\Farah\\Desktop\\police_tweet\\New folder")

out=args.output_dir


import os
import csv
def dd(input_topici,i):
    header=["time", "tweet"]
    ff=open(out+"topic_"+str(i)+".csv",'w',encoding='utf-8')
    k=csv.writer(ff, delimiter=',')
    k.writerow(['time','tweet'])

    for interval, f in enumerate(input_topici):
       
        print("Processing data from file " + f)
        inputf0 = open(args.input_dir+f,'r',encoding='utf-8')
        inputf = csv.reader(inputf0, delimiter = ',')
       
        for tweet in inputf:
            content = tweet[1]
            time=tweet[0]
            #print(content)
            k.writerow([time,content])
        inputf0.close()
    ff.close()
    
    sss=pd.read_csv(out+"topic_"+str(i)+".csv")   
    
    #r=sorted(sss['time'],key=lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S-04:00"))
    #sss['time']=r
    e=sss.values.tolist()
    sortedArray = sorted(e,key=lambda x: datetime.datetime.strptime(x[0], "%Y-%m-%d %H:%M:%S-04:00"))
    df = pd.DataFrame(sortedArray) 
    df.to_csv(out+"sorted_topic_"+str(i)+".csv",index=False)
       
num_topic=int(args.n)
a=[]
for i in range(num_topic):
    input_topici = [x for x in input_files if "topic"+str(i) in x]
    dd(input_topici,i)
    a.append(input_topici)



