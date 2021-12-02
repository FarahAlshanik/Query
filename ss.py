import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from os import listdir
from os.path import isfile, join

  
def get_lda_keywords(kwordsfile):
  #read in the top 5 keywords from the 5 topics 
  #return the list of list of keywords to expand queries
  #[["police","sdg","kw2"],..]
  ans = []
  inf = open(kwordsfile,'r')
  for topic in inf:
    keywords = set()
    words = topic.strip().split()
    keywords.update(words[:5])
    ans.append(keywords)
  return ans

def get_files(file_folder='',file_format='file_%d.csv',start_window=0):
  if len(file_folder):
    if file_folder[-1] != '/':
      file_folder += '/'
  
  num_files = len([f for f in listdir(file_folder) if isfile(join(file_folder, f))])
  files = []
  #from startwindow+1 to startwindow + 8 (2 hours window) 
  #for i in range(int(start_window), int(start_window+8)):
  # from start_window+1 to the end window
  for i in range(int(start_window), num_files):
    files.append(file_folder + file_format % (i + 1))
    print(files[-1])
  return files

def main(args):
  input_files = get_files(file_folder=args.query_input_files,start_window=args.start_window)
  lda_keywords = get_lda_keywords(kwordsfile=args.lda_top_20_keywords_file)

if __name__ == "__main__":
  # argument parse
  parser = argparse.ArgumentParser()
  parser.add_argument("--start_window",type=int, help = "start query from after this window",required=True) 
  args = parser.parse_args()
  main(args)
