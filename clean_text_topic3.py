import argparse
from os import listdir
from os.path import isfile, join
import sys
import dec_text_plda
import dec_graph
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

def get_files(file_folder='', file_format='topic3_file_%d.csv'): #file_format='file_%d.csv'
	"""
	Return all interval files in a given folder.
	Usage:
		#files = get_files('/zfs/dicelab/farah/query_exp/query-construction2/Boston/lda/lda_files', 'file_%d.csv')
		#files = get_files('/zfs/dicelab/farah/query_exp/query-construction2/Baltimore/lda_files/', 'file_%d.csv')
		files = get_files('/zfs/dicelab/farah/query_exp/query-construction2/Baltimore/step1_lda/topic3/', 'topic3_file_%d.csv')
		# files = ['hourly_intervals/file_1.csv' ... 'hourly_inverals/file_216.csv']
	:param file_folder: where your files are
	:param file_format: how your files are named
	:returns files: ordered list of files to process.
	"""
	if len(file_folder):
		if file_folder[-1] != '/':
			file_folder += '/'
	
	num_files = len([f for f in listdir(file_folder) if isfile(join(file_folder, f))])
#	print(listdir(file_folder))
	files = []
	for i in sorted(listdir(file_folder)):
		files.append(file_folder+i)
#		print(i)
#	print(files)
	return files


def build_args():
	"""
	Build out program arguments and return them.
	:returns: arguments pased by the user.
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument("--input_folder",\
			    help="Location of the folder where your (numbered/ordered) interval files are.",\
			    type=str,\
			    default="/zfs/dicelab/farah/query_exp/query-construction2/Baltimore/step1_lda/topic3/") #default="/zfs/dicelab/farah/query_exp/query-construction2/Baltimore/lda_files/

	parser.add_argument("--output_folder", help="location of output files", type=str, default='/zfs/dicelab/farah/query_exp/query-construction2/Baltimore/step1_lda/lda_files_topic3/') #/Baltimore/lda_clean_files/

	parser.add_argument("--P", help="Number of intervals to calculate DEC from.", type=int, default=5)

	parser.add_argument("--log_file", help="File to write output to.", type=str, default=None)
        
	# parser.add_argument("-v", "--verbosity", help="Prints various log messages", type=bool)

	return parser.parse_args()


def calculate_DEC(input_files=[], output_folder=''):
 
    files = []
    for i in ((input_files)):
       # files.append(file_folder+i)
        print(i[76:])
        files.append(i[76:])#should be different based on the path of files

    stopwords = dec_text_plda.getStopwords()
    #text=[]
    for interval, f in enumerate(input_files):
 #       print(f)
        print("Processing data from interval %d" % (interval + 1))


        text=( dec_text_plda.get_text_from_file(file=f, stopwords=stopwords))
        #print(text)        
        print(text[0])
    #outfile =  output_folder + 'file_%d.txt' % (interval + 1)
        outfile =  output_folder + files[interval]
    #csvfile2 = open(output_folder +  files[interval],'w', newline='')
    #f1 = csv.writer(csvfile2,delimiter='\n')
    #for item in text:

    
    #cc=text.split()
    #c = Counter(text.split())

    #for hotword in set(cc):
    #print (hotword, c[hotword])

    
    #f1.writerow(())

        #dec_text.write_dec_values(outfile=outfile, dec_vals=dec_vals, rank=True)
        ecentral = open(outfile,'w')
       # for i in range(5):
        #        x=lda.show_topic(i,topn=20)
        #ecentral.write(str(text))
          #      ecentral.write('\n')
        #ecentral.close()
        for item in text:
            cc=str(item).split()
            c=Counter(item.split())
            for h in set(cc):
            
                ecentral.write("%s" % h+" "+str(c[h])+ " ")
            ecentral.write("\n")
        ecentral.close()





if __name__ == "__main__":
	# argument parsing and some small format-checking
	args = build_args()

	if args.log_file:
	    sys.stdout = open(args.log_file, 'w')

	if len(args.output_folder):
		if args.output_folder[-1] != '/':
			args.output_folder += '/'
			print(args.output_folder)

	if len(args.input_folder):
		if args.input_folder[-1] != '/':
			args.input_folder += '/'
			print(args.input_folder)

	input_files = get_files(file_folder=args.input_folder)
#	print(input_files)
	calculate_DEC(input_files=input_files, output_folder=args.output_folder)
