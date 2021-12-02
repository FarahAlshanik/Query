#!/usr/bin/python
# Print a readable text of the model
# ./view_model.py model_file viewable_file
import sys, os
import argparse
from os import listdir
from os.path import isfile, join
import sys
import re
from pprint import pprint
import csv
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
from collections import Counter

def get_files(file_folder='', file_format='file_%d.csv'):
        """
        Return all interval files in a given folder.
        Usage:
                #files = get_files('/zfs/dicelab/farah/query_exp/query-construction2/Baltimore/topic_input/', 'file_%d.csv')
                #files = get_files('/zfs/dicelab/farah/query_exp/query-construction2/Baltimore/plda_output_new/', 'file_%d.csv')
        files = get_files('/zfs/dicelab/farah/query_exp/query-construction2/Baltimore/step1_lda//plda_output_topic4/', 'topic4_file_%d.csv')
	# files = ['hourly_intervals/file_1.csv' ... 'hourly_inverals/file_216.csv']
        :param file_folder: where your files are
        :param file_format: how your files are named
        :returns files: ordered list of files to process.
        """
        if len(file_folder):
                if file_folder[-1] != '/':
                        file_folder += '/'

        num_files = len([f for f in listdir(file_folder) if isfile(join(file_folder, f))])
#       print(listdir(file_folder))
        files = []
        for i in sorted(listdir(file_folder)):
                files.append(file_folder+i)
#               print(i)
#       print(files)
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
                            default="/zfs/dicelab/farah/query_exp/query-construction2/Baltimore/step1_lda/plda_output_topic4/") #Baltimore/plda_output_new/
        parser.add_argument("--output_folder", help="location of output files", type=str, default='/zfs/dicelab/farah/query_exp/query-construction2/Baltimore/step1_lda/plda_output_topic4_format/')

        parser.add_argument("--P", help="Number of intervals to calculate DEC from.", type=int, default=5)

        parser.add_argument("--log_file", help="File to write output to.", type=str, default=None)


        return parser.parse_args()


def calculate_DEC(input_files=[], output_folder=''):

    files = []
    for i in ((input_files)):
       # files.append(file_folder+i)
        print(i[88:])
        files.append(i[88:])#should be different based on the path of files
    for interval, f in enumerate(input_files):
        print("Processing data from interval %d" % (interval + 1))


        outfile =  output_folder + files[interval]
        ecentral = open(outfile,'w')
     
        num_topics = 0
        map = []
        sum = []
        word_sum = {}
        for line in open(f):
            sep = line.split("\t")
           # print(sep)
            word = sep[0]
         #   print(word)

            sep = sep[1].split()
            #print(sep)
            if num_topics == 0:
                num_topics = len(sep)
                for i in range(num_topics):
                    map.append({})
            for i in range(len(sep)):
		print(sep[i])
		if float(sep[i]) > 1:
                    map[i][word] = float(sep[i])
        for i in range(len(map)):
            for key in map[i].keys():
                map[i][key] = map[i][key]

        for i in range(len(map)):
            j=0
            x = sorted(map[i].items(), key=lambda(k, v):(v, k), reverse = True)
            for key in x:
                j+=1
                if(j==20):
                    break
   			#print key[0],
  			#print

                ecentral.write(key[0]+ " ")
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
#       print(input_files)
        calculate_DEC(input_files=input_files, output_folder=args.output_folder)







