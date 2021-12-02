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
			for word in content.strip().split():
				if word.lower.startswith("#"):
					hashtags[word] = hashtags.get(word,0)+1
		inputf0.close()
	sorted_hashtags = sorted(hashtags.items(), key = lambda kv: kv[1],reverse=True)	
	with open(output_file,'w') as outputf:
		for k,v in sorted_hashtags:
			outputf.write(k+" "+str(v)+"\n")

def count_hashtags_by_topic(input_files=[],output_file='',input_dir=''):
	#count number of hashtags for each topic for each time window
	hashtag_cnt_list=[]
	out_f = open(output_file,'a')
	for interval, f in enumerate(input_files):
		ht_cnt = 0
		print("Processing data from file " + f)
		inputf0 = open(os.path.join(input_dir,f),'r')
		inputf = csv.reader(inputf0, delimiter = ',')
		for tweet in inputf:
			content = tweet[1]
			for word in content.strip().split():
				if word.lower().startswith("#"):
					ht_cnt += 1
		inputf0.close()
		hashtag_cnt_list.append(str(ht_cnt))
	out_f.write('hashtag count: '+str(sum([int(cnt) for cnt in hashtag_cnt_list]))+'\n')
	out_f.write(','.join(hashtag_cnt_list)+"\n")
	out_f.close()
	

if __name__ == "__main__":
	# get input files
	input_files = dec_main.get_files(file_folder="/zfs/dicelab/farah/query_exp/query-construction2/Baltimore/interval/queryoutput/lda_int15/")#queryoutput
	parser = argparse.ArgumentParser()
	parser.add_argument("--input_dir",help="input directory of query result files to count hashtags", required=True)
	parser.add_argument("--output_file",help="output file for hashtags counting in query results", required=True)
	parser.add_argument("--num_topics",type=int,help="number of topics", required=True,default=5)
	args = parser.parse_args()
	inputdir = ('/zfs/dicelab/farah/query_exp/query-construction2/Baltimore/interval_all/int')#input files of police tweet
	input_files = os.listdir(args.input_dir)
	for i in range(args.num_topics):
		input_topici = [x for x in input_files if "topic"+str(i) in x]
		#output_topici = args.output_file+"_topic"+str(i)
		#generate_hashtags(input_files=input_topici,output_file=output_topici,input_dir=args.input_dir)
		count_hashtags_by_topic(input_files=input_topici,output_file=args.output_file,input_dir=args.input_dir)
