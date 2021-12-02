import argparse
import os
import numpy as np
import dec_text
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from os import listdir
from os.path import isfile, join
import nltk
nltk.download('wordnet')

def get_files_from_query_output(file_folder='',file_format='topic%d_file_%d.csv',start_window=0,num_topics=5):
  if len(file_folder):
    if file_folder[-1] != '/':
      file_folder += '/'
  
  num_windows = len([f for f in listdir(file_folder) if isfile(join(file_folder, f))]) //int(num_topics)
  files = []
  for t in range(num_topics):
    for i in range(int(start_window), int(start_window)+int(num_windows)):
      files.append(file_folder + file_format % (t,i + 1))
      print(files[-1])
  return files

def get_files(file_folder='',file_format='file_%d.csv',start_window=0):
  if len(file_folder):
    if file_folder[-1] != '/':
      file_folder += '/'
  
  num_files = len([f for f in listdir(file_folder) if isfile(join(file_folder, f))])
  files = []
  for i in range(int(start_window), int(start_window)+int(num_files)):
    files.append(file_folder + file_format % (i + 1))
    print(files[-1])
  return files

def main(args):
  #preprocess tweets raw data and generate raw format for word embedding learning
  stopwords = dec_text.getStopwords()
  #to get query result for first experiment, use window 16 as start window
  input_files = get_files(file_folder=args.query_input_files,start_window=0)
 # input_files = get_files_from_query_output(file_folder=args.query_input_files,start_window=16) # you should change start window based on the window for each query result
  output_file = args.preprocess_output_file
  outf = open(output_file,'w')
  for input_file in input_files:
    text = dec_text.get_text_from_file(file=input_file)
    for line in text:
      outf.write(' '.join(line)+"\n")
  outf.close()

if __name__ == "__main__":
  # argument parse
  parser = argparse.ArgumentParser()
  parser.add_argument("--query_input_files", help = "directory contains files to start query on",required=True) 
  parser.add_argument("--preprocess_output_file",help="name of output file for word embedding generation ",required=True)
  args = parser.parse_args()
  main(args)
