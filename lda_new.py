import argparse
from os import listdir
from os.path import isfile, join
import sys
import dec_text
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
from collections import Counter
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import re

def get_files(file_folder='', file_format='file_%d.csv'):
	"""
	Return all interval files in a given folder.
	Usage:
		files = get_files('/zfs/dicelab/farah/Baltimore/lda_files', 'file_%d.csv')
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
		print(files[-1])
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
			    default="/zfs/dicelab/farah/Baltimore/lda_files")

	parser.add_argument("--output_folder", help="location of output files", type=str, default='/zfs/dicelab/farah/Baltimore/lda_output')

	parser.add_argument("--P", help="Number of intervals to calculate DEC from.", type=int, default=5)

	parser.add_argument("--log_file", help="File to write output to.", type=str, default=None)
        
	# parser.add_argument("-v", "--verbosity", help="Prints various log messages", type=bool)

	return parser.parse_args()


def calculate_DEC(input_files=[], output_folder=''):
 
    stopwords = dec_text.getStopwords()

    for interval, f in enumerate(input_files):
        print(f)
        a=re.findall(r'\d+', f)
        num=(int(a[0])) 
        print("Processing data from interval %d" % (interval + 1))
 

        text = dec_text.get_text_from_file(file=f, stopwords=stopwords)
#        print(text)        
        id2word = corpora.Dictionary(text)

        # Create Corpus
        texts = text

        # Term Document Frequency
        corpus = [id2word.doc2bow(t) for t in texts]
        lda = LdaMulticore(corpus, id2word=id2word, num_topics=5,random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           per_word_topics=True)

  #      pprint(lda.print_topics())                               
#        for i in range(5):
 #               x=lda.show_topic(i, topn=20)
             #   for i in x:
  #              print(x)
        outfile = output_folder + 'file_%d.txt' % (num)
       # dec_text.write_dec_values(outfile=outfile, dec_vals=dec_vals, rank=True)
        ecentral = open(outfile,'w')
        for i in range(5):
                x=lda.show_topic(i,topn=20)
#                print(x)
                for item in x:
                    cc=item[0]
                    ecentral.write("%s"% str(cc)+ " ")
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
	print(input_files)
	calculate_DEC(input_files=input_files, output_folder=args.output_folder)
