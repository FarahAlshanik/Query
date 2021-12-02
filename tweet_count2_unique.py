
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from os import listdir
from os.path import isfile, join
import math
import re

import csv, subprocess
import time
import os
import sys
import argparse
from os import listdir

import dec_main

def generate_hashtags(input_files=[],output_file='',input_dir=''):
    #count number of hashtags from query results
        hashtags = dict()
        for interval, f in enumerate(input_files):
                print("Processing data from file " + f)
                inputf0 = open(os.path.join(input_dir,f),'r')
                inputf = csv.reader(inputf0, delimiter = ',')
                for tweet in inputf:
                        content = tweet[1]
                        content=re.sub(r'[^#\w\s]', '', content)
                        for word in content.strip().split():
                                if word.lower().startswith("#"):
                                        hashtags[word.lower()] = hashtags.get(word.lower(),0)+1
                inputf0.close()
        sorted_hashtags = sorted(hashtags.items(), key = lambda kv: kv[1],reverse=True)
        with open(output_file,'w') as outputf:
                for k,v in sorted_hashtags:
                        outputf.write(k+" "+str(v)+"\n")


'''
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
                         #       if word.lower()==keyword:
                          #              ht_cnt += 1
                                        
                inputf0.close()
                hashtag_cnt_list.append(str(ht_cnt))
        out_f.write('tweet count: '+str(sum([int(cnt) for cnt in hashtag_cnt_list]))+'\n')
        out_f.write(','.join(hashtag_cnt_list)+"\n")
        out_f.close()
'''
def get_files(file_folder=''):
        if len(file_folder):
                if file_folder[-1] != '/':
                        file_folder += '/'

        num_files = len([f for f in listdir(file_folder) if isfile(join(file_folder, f))])

        files = [f for f in listdir(file_folder) if isfile(join(file_folder, f))]
        print(files)
        #for i in range(0, num_files):
         #       files.append(file_folder + file_format % (i + 1))
          #      print(files[-1])

        return files



if __name__ == "__main__":
        # get input files
        #input_files = dec_main.get_files(file_folder="/zfs/dicelab/farah/query_exp/query-construction2/Baltimore/interval/queryoutput/lda_int15/")#queryoutput
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_files",help="input directory of all tweets",required=True)
        parser.add_argument("--input_dir",help="input directory of query result files to count hashtags", required=True)
        parser.add_argument("--output_file",help="output file for hashtags counting in query results", required=True)
        parser.add_argument("--num_topics",type=int,help="number of topics", required=True,default=5)
        args = parser.parse_args()

        input_files=get_files(file_folder=args.input_files)

        ##inputdir = ('/zfs/dicelab/farah/query_exp/query-construction2/Baltimore/interval_all/int')#input files of police tweet
        input_files_all = os.listdir(args.input_dir) #query output files will be  input diretory to count hashtag
        for i in range(args.num_topics):
                input_topici = [x for x in input_files if "topic"+str(i) in x]
                print(input_topici)
                output_topici = args.output_file+"_topic"+str(i)
                generate_hashtags(input_files=input_topici,output_file=output_topici,input_dir=args.input_dir)
                #count_hashtags_by_topic(input_files=input_topici,output_file=args.output_file,input_dir=args.input_files)











'''
if __name__ == "__main__":
  # argument parse
  parser = argparse.ArgumentParser()
  #parser.add_argument("--lda_top_20_keywords_file", help = "top 20 keywords for each topic file",required=True)
  #parser.add_argument("--dec_keyword_file", help = "top dec keywords",required=False)
  #parser.add_argument("--word_vectors_voc_file", help = "file of look up table of word embeddings and dec/lda keywords",required=True)
  parser.add_argument("--query_input_files", help = "directory contains files to start query on",required=True)
  #parser.add_argument("--start_window",type=int, help = "start query from after this window",required=True)
  #parser.add_argument("--n", type=int,help = "number of keywords to use for query",required=True)
  #parser.add_argument("--query_output_directory",help="directory to save query result files",required=True)
  parser.add_argument("--output_file",help="file to save tweet count for each topic",required=True)
  parser.add_argument("--num_topics",type=int, help = "number of topic",required=True)
  parser.add_argument("--keyword", help = "hashtag_keyword",required=True)
  args = parser.parse_args()
  input_files=get_files(file_folder=args.query_input_files)

  input_files_all = os.listdir(args.query_input_files) #query output files will be  input diretory to count hashtag
  for i in range(args.num_topics):
      input_topici = [x for x in input_files if "topic"+str(i) in x]
      print(input_topici)
      output_topici = args.output_file+"_topic"+str(i)
      generate_hashtags(input_files=input_topici,output_file=output_topici,input_dir=args.query_input_files)
      #count_tweets_hashtags_by_topic(input_files=input_topici,output_file=args.output_file,input_dir=args.query_input_files,keyword=args.keyword)

  #main(args)


'''
