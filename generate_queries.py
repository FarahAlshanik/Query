import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from os import listdir
from os.path import isfile, join
import math

def Remove(sets):
    if("police" in sets):
        sets.remove("police")
    if("rt" in sets):
        sets.remove("rt")



def distance(word_vec,word1,word2):
  emd1,emd2 = word_vec[word1],word_vec[word2]
#  return sum([(a-b)**2 for a,b in zip(emd1,emd2)])
  return math.sqrt(sum([(a - b) ** 2 for a, b in zip(emd1, emd2)]))


def read_wordembedding(vocfile):
  word_vectors = {}
  voc = open(vocfile,'r')
  for line in voc:
    word,embedding = line.strip().split(' ',1)  
    embedding = [float(e) for e in embedding.split()]
    word_vectors[word]=embedding
  voc.close()
  return word_vectors

#visualize word embeddings
def plot_wordembedding(vocfile):
  words = []
  vecs = []
  voc = open(vocfile,'r')
  for line in voc:
    word,embedding = line.strip().split(' ',1)  
    embedding = [float(e) for e in embedding.split()]
    words.append(word)
    vecs.append(embedding)
  voc.close()
  U,s,Vh = np.linalg.svd(vecs, full_matrices=False)
  colors = ['red','green','blue','yellow','orange','black']
  for i in range(len(words)):
    fig = plt.gcf()
    fig.set_size_inches(18.5,10.5)
    colori = colors[i//20] if i < 100 else colors[5]
    plt.text(U[i,0],U[i,1],words[i],color=colori)
    plt.xlim((-0.5,0.5))
    plt.ylim((-0.5,0.5))
  plt.savefig('viz_dec_lda_words.jpg')

def append_dec_keywords(decfile,ldakeywords,vocfile):
  #return the list of list of keywords to expand queries
  #for each topic's top 5 keywords, append the dec word that has the maximum vector spa
  # distance from them. 
  # return [[decword1, topic1word1, ...topic1word5],...]
  dec_25 = set()
  dec_words = open(decfile,'r')
  word_vectors = read_wordembedding(vocfile)
  #read in 25 dec keywords
  count = 5
  for line in dec_words:
    word = line.strip().split()[0]
    #if(word not in ldakeywords)
    dec_25.add(word)
    count -= 1
    if count == 0:
      break
  ans = []
  #read in word vectors
  for topic in ldakeywords:
    maxdist = -1.0
    dec_select_word = ''
    for dec_word in dec_25:
      #if(dec_word in topic):
       # continue
      # find the dec_word that has maximum distance with top 5 lda keywords
      dist = sum([distance(word_vectors,word,dec_word) for word in topic])
      if dist >= maxdist:
        dec_select_word=dec_word
        maxdist = dist
        print (dec_word + " distance "+ str(dist))
    ans.append([dec_select_word]+list(topic))
  print (ans)
  dec_words.close()
  return ans
  
def get_lda_keywords(kwordsfile):
  #read in the top 5 keywords from the 5 topics 
  #return the list of list of keywords to expand queries
  #[["police","sdg","kw2"],..]
  ans = []
  inf = open(kwordsfile,'r')
  for topic in inf:
    keywords = set()
    words = topic.strip().split()
    Remove(words)
    keywords.update(words[:18])
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
  #generate 5 queries on each of the 5 topics
  input_files = get_files(file_folder=args.query_input_files,start_window=args.start_window)
  #plot word embeddings 
  #plot_wordembedding(vocfile=args.word_vectors_voc_file)
  #keywords from lda topics
  lda_keywords = get_lda_keywords(kwordsfile=args.lda_top_20_keywords_file)
  #keywords from dec
  keywords = append_dec_keywords(decfile=args.dec_keyword_file,ldakeywords=lda_keywords,vocfile=args.word_vectors_voc_file)
  num_topics = len(keywords)
  #tweet count output file
  tweet_cnt_out_f = open(args.tweet_cnt_output_file,'w')
  for topic_id in range(num_topics):
    tweet_cnt_list = []
    for input_file in input_files:
      input_dir,basename = os.path.split(input_file)
      inf = open(input_file,'r')
      output_file = os.path.join(args.query_output_directory,"topic"+str(topic_id)+"_"+basename)
      outf = open(output_file,'w')
      tweet_cnt = 0
      for line in inf:
        cnt = 0
        #must contains dec_word and aleast n keywords from lda keywords
       ## if keywords[topic_id][0] not in line.lower():
         ## continue
        for word in keywords[topic_id][0:]: ##change from 1 to 0
        #for word in lda_keywords[topic_id][0]:   #add
          if word in line.lower():
            cnt += 1 
          if cnt >=int(args.n):
            #outf = open(output_file,'w')

            # if contains at least n keywords, then write to output  
            outf.write(line)
            tweet_cnt += 1
            break
      outf.close()
      inf.close()
      tweet_cnt_list.append(str(tweet_cnt))
      sum_tweet = sum([int(cnt) for cnt in tweet_cnt_list])
    #flush tweet cnt of a topic to file
    tweet_cnt_out_f.write('topic '+str(topic_id)+" tweet count: "+str(sum_tweet)+'\n')
    tweet_cnt_out_f.write(','.join(tweet_cnt_list)+'\n')
  tweet_cnt_out_f.close()

if __name__ == "__main__":
  # argument parse
  parser = argparse.ArgumentParser()
  parser.add_argument("--lda_top_20_keywords_file", help = "top 20 keywords for each topic file",required=True) 
  parser.add_argument("--dec_keyword_file", help = "top dec keywords",required=False) 
  parser.add_argument("--word_vectors_voc_file", help = "file of look up table of word embeddings and dec/lda keywords",required=True) 
  parser.add_argument("--query_input_files", help = "directory contains files to start query on",required=True) 
  parser.add_argument("--start_window",type=int, help = "start query from after this window",required=True) 
  parser.add_argument("--n", type=int,help = "number of keywords to use for query",required=True) 
  parser.add_argument("--query_output_directory",help="directory to save query result files",required=True)
  parser.add_argument("--tweet_cnt_output_file",help="file to save tweet count for each topic",required=True)
  args = parser.parse_args()
  main(args)
