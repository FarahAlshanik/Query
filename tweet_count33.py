
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from os import listdir
from os.path import isfile, join
import math


import csv, subprocess
import time
import os
import sys
import argparse
from os import listdir

import dec_main



def get_files(file_folder='',file_format='file_%d.csv',start_window=0):
  if len(file_folder):
    if file_folder[-1] != '/':
      file_folder += '/'

  num_files = len([f for f in listdir(file_folder) if isfile(join(file_folder, f))])
  files = []
  #from startwindow+1 to startwindow + 8 (2 hours window)
  #for i in range(int(start_window), int(start_window+8)):
  # from start_window+1 to the end window
  for i in range(int(start_window)-1, num_files):
    files.append(file_folder + file_format % (i + 1))
    #print(files[-1])
  print(files)
  return files




def generate_hashtags(input_files=[],output_file='',input_dir=''):
    #count number of hashtags from query results
        hashtags = dict()
        for interval, f in enumerate(input_files):
                print("Processing data from file " + f)
                inputf0 = open(os.path.join(input_dir,f),'r')
                inputf = csv.reader(inputf0, delimiter = ',')
                for tweet in inputf:
                        content = tweet[1]
                        for word in content.strip().split():
                                if word.lower.startswith("#"):
                                        hashtags[word] = hashtags.get(word,0)+1
                inputf0.close()
        sorted_hashtags = sorted(hashtags.items(), key = lambda kv: kv[1],reverse=True)
        with open(output_file,'w') as outputf:
                for k,v in sorted_hashtags:
                        outputf.write(k+" "+str(v)+"\n")

def count_tweets_hashtags_by_topic(input_files=[],output_file='',input_dir='',keyword=''):
        #count number of hashtags for each topic for each time window
        hashtag_cnt_list=[]
        out_f = open(output_file,'a')
        for interval, f in enumerate(input_files):
                ht_cnt = 0
                print("Processing data from file " + f)
                inputf0 = open(os.path.join(input_dir,f),'r')
                inputf = csv.reader(inputf0, delimiter = ',')
                for tweet in inputf:
                        content = tweet[1].lower()
                        if(keyword in content):
                            ht_cnt += 1
                        #for word in content.strip().split():
                                #if word.lower()==keyword:
                                        #ht_cnt += 1
                                        
                inputf0.close()
                hashtag_cnt_list.append(str(ht_cnt))
        out_f.write('tweet count: '+str(sum([int(cnt) for cnt in hashtag_cnt_list]))+'\n')
        out_f.write(','.join(hashtag_cnt_list)+"\n")
        out_f.close()




if __name__ == "__main__":
  # argument parse
  parser = argparse.ArgumentParser()
  #parser.add_argument("--lda_top_20_keywords_file", help = "top 20 keywords for each topic file",required=True)
  #parser.add_argument("--dec_keyword_file", help = "top dec keywords",required=False)
  #parser.add_argument("--word_vectors_voc_file", help = "file of look up table of word embeddings and dec/lda keywords",required=True)
  parser.add_argument("--query_input_files", help = "directory contains files to start query on",required=True)
  parser.add_argument("--start_window",type=int, help = "start query from after this window",required=True)
  #parser.add_argument("--n", type=int,help = "number of keywords to use for query",required=True)
  #parser.add_argument("--query_output_directory",help="directory to save query result files",required=True)
  parser.add_argument("--output_file",help="file to save tweet count for each topic",required=True)
  #parser.add_argument("--num_topics",type=int, help = "number of topic",required=True)
  #parser.add_argument("--keyword", help = "hashtag_keyword",required=True)
  #keywords=['#looting','#protest','violence']
  parser.add_argument("--keyword", help = "hashtag_keyword",required=True)

  args = parser.parse_args()
  input_files=get_files(file_folder=args.query_input_files,start_window=args.start_window)

  #input_files_all = input_files #query output files will be  input diretory to count hashtag
  input_topici = input_files
  output_topici = args.output_file+'_'+args.keyword
  #generate_hashtags(input_files=input_topici,output_file=output_topici,input_dir=args.input_dir)
  count_tweets_hashtags_by_topic(input_files=input_topici,output_file=args.output_file,input_dir=args.query_input_files,keyword=args.keyword)

  #main(args)


