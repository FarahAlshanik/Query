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


def build_args():
	"""
	Build out program arguments and return them.
	:returns: arguments pased by the user.
	"""
	parser = argparse.ArgumentParser()

	parser.add_argument("--output_folder", help="location of output files", type=str, default='/zfs/dicelab/farah/query_exp/query-construction2/Baltimore/')

	parser.add_argument("--P", help="Number of intervals to calculate DEC from.", type=int, default=5)

	parser.add_argument("--log_file", help="File to write output to.", type=str, default=None)
        
	# parser.add_argument("-v", "--verbosity", help="Prints various log messages", type=bool)

	return parser.parse_args()


def calculate_DEC( output_folder=''):
 
    #i=0
    stopwords = dec_text_plda.getStopwords()
    files ='/zfs/dicelab/farah/query_exp/query-construction2/Baltimore/sorted.csv'

    text=( dec_text_plda.get_text_from_file(file=files, stopwords=stopwords))
   # print(text)
    outfile =  output_folder + 'sorted_clean.csv'
    ecentral = open(outfile,'w')
    for item in text:
            
            ecentral.write(str(item))
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


	calculate_DEC( output_folder=args.output_folder)
