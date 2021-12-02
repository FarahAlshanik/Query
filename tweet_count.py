
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from os import listdir
from os.path import isfile, join
import math



def get_files(file_folder='',file_format='file_%d.csv',start_window=0):
  if len(file_folder):
    if file_folder[-1] != '/':
      file_folder += '/'

  num_files = len([f for f in listdir(file_folder) if isfile(join(file_folder, f))])
  files = []
  #from startwindow+1 to startwindow + 8 (2 hours window)
  #for i in range(int(start_window), int(start_window+8)):
  # from start_window+1 to the end window
  for i in range(int(start_window)-1, num_files):
    files.append(file_folder + file_format % (i + 1))
    print(files[-1])
  return files


def main(args):
  #generate 5 queries on each of the 5 topics
  input_files = get_files(file_folder=args.query_input_files,start_window=args.start_window)
  #plot word embeddings
  #plot_wordembedding(vocfile=args.word_vectors_voc_file)
  #keywords from lda topics
  #lda_keywords = get_lda_keywords(kwordsfile=args.lda_top_20_keywords_file)
  #keywords from dec
  #keywords = append_dec_keywords(decfile=args.dec_keyword_file,ldakeywords=lda_keywords,vocfile=args.word_vectors_voc_file)
  keywords=['#looting','#protest']
  num_topics = 5
  sum_tweet=0
  #tweet count output file
  tweet_cnt_out_f = open(args.tweet_cnt_output_file,'w')
  for topic_id in range(num_topics):
    tweet_cnt_list = []
    for input_file in input_files:
      input_dir,basename = os.path.split(input_file)
      inf = open(input_file,'r')
      #output_file = os.path.join(args.query_output_directory,"topic"+str(topic_id)+"_"+basename)
      #outf = open(output_file,'w')
      tweet_cnt = 0
      for line in inf:
        cnt = 0
        #must contains dec_word and aleast n keywords from lda keywords
       ## if keywords[topic_id][0] not in line.lower():
         ## continue
        for word in keywords: ##change from 1 to 0
        #for word in lda_keywords[topic_id][0]:   #add
          if word in line.lower():
            cnt += 1
          #if cnt >=int(args.n):
            #outf = open(output_file,'w')

            # if contains at least n keywords, then write to output
            #outf.write(line)
            tweet_cnt += 1
            break
      #outf.close()
      inf.close()
      tweet_cnt_list.append(str(tweet_cnt))
      sum_tweet = sum([int(cnt) for cnt in tweet_cnt_list])

    tweet_cnt_out_f.write('topic '+str(topic_id)+" tweet count: "+str(sum_tweet)+'\n')
    tweet_cnt_out_f.write(','.join(tweet_cnt_list)+'\n')
  tweet_cnt_out_f.close()

if __name__ == "__main__":
  # argument parse
  parser = argparse.ArgumentParser()
  #parser.add_argument("--lda_top_20_keywords_file", help = "top 20 keywords for each topic file",required=True)
  #parser.add_argument("--dec_keyword_file", help = "top dec keywords",required=False)
  #parser.add_argument("--word_vectors_voc_file", help = "file of look up table of word embeddings and dec/lda keywords",required=True)
  parser.add_argument("--query_input_files", help = "directory contains files to start query on",required=True)
  parser.add_argument("--start_window",type=int, help = "start query from after this window",required=True)
  #parser.add_argument("--n", type=int,help = "number of keywords to use for query",required=True)
  #parser.add_argument("--query_output_directory",help="directory to save query result files",required=True)
  parser.add_argument("--tweet_cnt_output_file",help="file to save tweet count for each topic",required=True)
  args = parser.parse_args()
  main(args)


