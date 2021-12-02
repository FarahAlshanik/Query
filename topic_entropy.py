import csv, subprocess
import time
import os
import sys
import argparse
from os import listdir

import dec_main
import dec_text

STOPWORDS = dec_text.getStopwords()

def compute_entropy(input_files=[],input_dir='',freqcount={}):
	"""
	compute entropy for a query result related to topic i 
	:return average entropy per tweet of the input_files
	"""
	ans = []
	for interval, f in enumerate(input_files):
		entropy = 0.0
		tweet_cnt = 0
		print("Processing data from file " + f)
		inputf0 = os.path.join(input_dir,f)
		text = dec_text.get_text_from_file(file=inputf0,stopwords=STOPWORDS)
		tweet_cnt += len(text)
		entropyi = dec_text.getEntropy(text=text,freqcount=freqcount)
		entropy += entropyi
		#avg_entropy per tweet
		avg_entropy = 0.0 if tweet_cnt == 0 else entropy / float(tweet_cnt) 
		ans.append(avg_entropy)
	#avg_entropy = 0.0 if tweet_cnt == 0 else entropy / float(tweet_cnt) 
	#return avg_entropy
	return ','.join([ str(e) for e in ans ])
	

if __name__ == "__main__":
	# get input files
	#input_files = dec_main.get_files(file_folder="/scratch2/yuhengd/boston/interval/queryoutput/int15")
	parser = argparse.ArgumentParser()
	parser.add_argument("--input_dir",help="input directory of query result files to count hashtags", required=True)
	parser.add_argument("--output_file",help="output file for entropy measure in query results", required=True)
	parser.add_argument("--num_topics",type=int,help="number of topics", required=True)
	parser.add_argument("--wordcount_input_file",help="file to perform word count", required=True)
	args = parser.parse_args()
	input_files = os.listdir(args.input_dir)
	output_f = open(args.output_file,'w')
	#get word count map and save it in freqcount
	wordcount_file = args.wordcount_input_file
	freqcount = dec_text.getWordCount(file=wordcount_file)
	#measure entropy of each topic's query results
	for i in range(args.num_topics):
		input_topici = [x for x in input_files if "topic"+str(i) in x]
		#output_topici = args.output_file+"_topic"+str(i)
		entropy = compute_entropy(input_files=input_topici,input_dir=args.input_dir,freqcount=freqcount)
		output_f.write("entropy of query results from topic "+str(i) + " is "+"\n"+str(entropy)+"\n")
	output_f.close()
