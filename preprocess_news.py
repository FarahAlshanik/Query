import argparse
import os
import numpy as np
import dec_text
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from os import listdir
from os.path import isfile, join


def main(args):
  #preprocess tweets raw data and generate raw format for word embedding learning
  stopwords = dec_text.getStopwords()
  #to get query result for first experiment, use window 16 as start window
  input_file = args.query_input_file
  output_file = args.preprocess_output_file
  outf = open(output_file,'w')
  text = dec_text.get_text_from_news_file(file=input_file,stopwords=stopwords)
  for line in text:
    outf.write(' '.join(line)+'\n')
  outf.close()

if __name__ == "__main__":
  # argument parse
  parser = argparse.ArgumentParser()
  parser.add_argument("--query_input_file", help = "directory contains files to start query on",required=True) 
  parser.add_argument("--preprocess_output_file",help="name of output file for word embedding generation ",required=True)
  args = parser.parse_args()
  main(args)
