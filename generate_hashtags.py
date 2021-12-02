import csv, subprocess
import time
import sys

import dec_main

def generate_hashtags(input_files=[],output_file=''):
	hashtags = dict()
	for interval, f in enumerate(input_files):
		print("Processing data from interval %d" % (interval + 1))
		inputf0 = open(f,'r')
		inputf = csv.reader(inputf0, delimiter = ',')
		for tweet in inputf:
			content = tweet[1]
			for word in content.strip().split():
				if word.startswith("#"):
					hashtags[word] = hashtags.get(word,0)+1
		inputf0.close()
	sorted_hashtags = sorted(hashtags.items(), key = lambda kv: kv[1],reverse=True)	
	with open(output_file,'w') as outputf:
		for k,v in sorted_hashtags:
			outputf.write(k+" "+str(v)+"\n")

if __name__ == "__main__":
	# get input files
	input_files = dec_main.get_files(file_folder="boston/intervals/int")
	output_file = "hashtags_int15.txt"
	generate_hashtags(input_files=input_files,output_file=output_file)
