import argparse
import os
import numpy as np
from os import listdir
from os.path import isfile, join
from pyfasttext import FastText
import higest_freq_param2



def Remove(sets):
    if("police" in sets):
        sets.remove("police")
    if("rt" in sets):
        sets.remove("rt")



def distance(word_vec,word1,word2):
  emd1,emd2 = word_vec[word1],word_vec[word2]
  return sum([(a-b)**2 for a,b in zip(emd1,emd2)])

def read_wordembedding(vocfile):
  word_vectors = {}
  voc = open(vocfile,'r')
  for line in voc:
    word,embedding = line.strip().split(' ',1)  
    embedding = [float(e) for e in embedding.split()]
    word_vectors[word]=embedding
  voc.close()
  return word_vectors



def get_lda_keywords(kwordsfile,lda_thr):
  #read in the top 5 keywords from the 5 topics
  #return the list of list of keywords to expand queries
  #[["police","sdg","kw2"],..]
  ans = []
  inf = open(kwordsfile,'r')
  for topic in inf:
    keywords = set()
    words = topic.strip().split()
    Remove(words)
    keywords.update(words[:lda_thr])
    ans.append(keywords)
  return ans


def get_news_nn_keywords(lda_words,number):
  #read in top 5 keywordes from 5 topics, find 2 nn in news space of each keyword and return news nn keywords
  ans=[]
  #nottopicn=0
  #intopicn=0
  words_topic=[]
  words_nn=[]
  #ss=[]
  #model = FastText('./cnn_2014.bin')
  #model = FastText('/zfs/dicelab/farah/model_skip_300.bin')
  print ("lda keywords:")
  print(lda_words)
  for topic in lda_words:
    topic_nn = set()
    for word in topic:
      words_topic.append(word)
      word_nn=higest_freq_param2.find_words(word,number)
      #copy_word_nn=word_nn
      words_nn.append(word_nn)
      #ss.append(word_nn)
      print('lda word: ',word,'  words in nn:',word_nn)
      #if(word in word_nn): ##high freq words in topic 
       # intopicn+=1
      #else: ##high freq word not in topic
       # nottopicn+=1
      #word_nn = model.nearest_neighbors(word,k=2)
      #word_nn = [ t[0] for t in word_nn]
      word_nn.append(word)
      topic_nn.update(word_nn)
    #print('number of words not in topic:',nottopicn)
    #print('number of words in topic:',intopicn)
    ans.append(topic_nn)
  print ("historical space nn:")
  print (ans)
  #return ans,nottopicn,intopicn
  print('words_topic:',words_topic)
  #print('words_nn',words_nn)

  flat_list = [item for sublist in words_nn for item in sublist[0:len(sublist)-1]]
  words_nn_new=flat_list

  list3 = set(words_nn_new)&set(words_topic)  #intersection between 2 lists mean words from lda is in nn 
  list4 = sorted(list3, key = lambda k : words_nn_new.index(k))
  intopicn=len(list4) #numbr of  words in topics from nn
  nottopicn=len(words_nn_new)-len(list4)  #number of words in nn but not in topics
  print('words_nn',words_nn_new)
  return ans,words_topic,words_nn_new,intopicn,nottopicn

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
  lda_keywords = get_lda_keywords(kwordsfile=args.lda_top_20_keywords_file,lda_thr=int(args.lda))
  #keywords from news space nn
  news_nn_keywords,wt,wnn,intopicn,nottopicn = get_news_nn_keywords(lda_keywords,number=int(args.number))
  print('number of words not in topic=',nottopicn)
  print('number of words in topic=',intopicn)
  print(wnn)
  num_topics = len(lda_keywords)
  #tweet count output file
  tweet_cnt_out_f = open(args.tweet_cnt_output_file,'w')
  tweet_statistics=open('/zfs/dicelab/farah/tweet_stat.txt','w')
  tweet_statistics.write("number of words not in topic: "+str(nottopicn)+"    ")
  tweet_statistics.write("************************************************************************")
  tweet_statistics.write("number of words in topic: "+str(intopicn)+"     ")
  tweet_statistics.write("*************************************************************************")
  
  intweetn=0
  notintweet=0
  for topic_id in range(num_topics):
    #tweet_cnt_list = []
    for input_file in input_files:
      input_dir,basename = os.path.split(input_file)
      inf = open(input_file,'r')
      output_file = os.path.join(args.query_output_directory,"topic"+str(topic_id)+"_"+basename)
      #outf = open(output_file,'w')
      #tweet_cnt = 0
      for line in inf:
        #cnt = 0
        for word in wnn:

          if word in line.lower():
            intweetn+=1 
            #cnt += 1 
          else:
            notintweet+=1
          #if cnt >=int(args.n):
            # if contains at least n keywords, then write to output  
           # outf.write(line)
            #tweet_cnt += 1
            #break
      #outf.close()
      #inf.close()
      #tweet_cnt_list.append(str(tweet_cnt))
      #sum_tweet = sum([int(cnt) for cnt in tweet_cnt_list])
    #flush tweet cnt of a topic to file
    #tweet_cnt_out_f.write('topic '+str(topic_id)+" tweet count: "+str(sum_tweet)+'\n')
    #tweet_cnt_out_f.write(','.join(tweet_cnt_list)+'\n')
  print("number of words not in tweets: "+str(notintweet)+"    ")
  print("number of words in tweets: "+str(intweetn)+"    ")
  tweet_statistics.write("number of words not in tweets: "+str(notintweet)+"    ")
  tweet_statistics.write("************************************************************************")
  tweet_statistics.write("number of words in tweets: "+str(intweetn)+"    ")
  tweet_statistics.write("**************************************************************************")
  tweet_statistics.close()
  #tweet_cnt_out_f.close()

if __name__ == "__main__":
  # argument parse
  parser = argparse.ArgumentParser()
  parser.add_argument("--lda", help = "lda threshold between 5 and 20",required=True)
  parser.add_argument("--number",help="number of highest freq words", required=True)

  parser.add_argument("--lda_top_20_keywords_file", help = "top 20 keywords for each topic file",required=True) 
  parser.add_argument("--query_input_files", help = "directory contains files to start query on",required=True) 
  parser.add_argument("--start_window",type=int, help = "start query from after this window",required=True) 
  parser.add_argument("--n", type=int,help = "number of keywords to use for query",required=True) 
  parser.add_argument("--query_output_directory",help="directory to save query result files",required=True)
  parser.add_argument("--tweet_cnt_output_file",help="file to save tweet count for each topic",required=True)
  args = parser.parse_args()
  main(args)
