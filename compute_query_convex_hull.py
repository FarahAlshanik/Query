import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from os import listdir
from os.path import isfile, join
#from scipy.spatial import ConvexHull

from collections import namedtuple  
import matplotlib.pyplot as plt  
import random

Point = namedtuple('Point', 'x y')


class ConvexHull(object):  
  _points = []
  _hull_points = []
  _words = {}
  _hull_words = []

  def __init__(self):
    pass

  def add(self, point, word):
    self._points.append(point)
    self._words[point]=word
    

  def _get_orientation(self, origin, p1, p2):
    '''
    Returns the orientation of the Point p1 with regards to Point p2 using origin.
    Negative if p1 is clockwise of p2.
    :param p1:
    :param p2:
    :return: integer
    '''
    difference = (
            ((p2.x - origin.x) * (p1.y - origin.y))
            - ((p1.x - origin.x) * (p2.y - origin.y))
    )

    return difference

  def compute_hull(self):
    '''
    Computes the points that make up the convex hull.
    :return:
    '''
    points = self._points

    # get leftmost point
    start = points[0]
    min_x = start.x
    for p in points[1:]:
      if p.x < min_x:
        min_x = p.x
        start = p

    point = start
    self._hull_points.append(start)

    far_point = None
    while far_point is not start:

      # get the first point (initial max) to use to compare with others
      p1 = None
      for p in points:
        if p is point:
          continue
        else:
          p1 = p
          break

      far_point = p1

      for p2 in points:
        # ensure we aren't comparing to self or pivot point
        if p2 is point or p2 is p1:
          continue
        else:
          direction = self._get_orientation(point, far_point, p2)
          if direction > 0:
            far_point = p2

      self._hull_points.append(far_point)
      point = far_point

  def get_hull_points(self):
    if self._points and not self._hull_points:
      self.compute_hull()

    return self._hull_points

  def display(self):
    # all points
    x = [p.x for p in self._points]
    y = [p.y for p in self._points]
    plt.plot(x, y, marker='D', linestyle='None')

    # hull points
    hx = [p.x for p in self._hull_points]
    hy = [p.y for p in self._hull_points]
    plt.plot(hx, hy)
    for i in range(len(hx)):
      plt.text(hx[i],hy[i],self._words[self._hull_points[i]])

    plt.title('Convex Hull')
    #plt.show()
    plt.savefig('convex_hull_news_nn3_v2.jpg')


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

def plot_wordembedding(vocfile):
  full_vec={}
  full_vec_file = open('tweet_space.vec_pca_embed.txt','r')
  for line in full_vec_file:
    word,embedding = line.strip().split(' ',1)  
    embedding = [float(e) for e in embedding.split()]
    if isinstance(word,str):
      full_vec[word] = embedding
  full_vec_file.close()
  print('number of words in full vec: '+str(len(full_vec.keys())))
  words = []
  wordset= set()
  vecs = []
  voc = open(vocfile,'r')
  oov_set = set()
  cnt1,cnt2=0,0
  for line in voc:
    cnt1+=1
    word = line.strip().split(' ')[0]
    if word not in full_vec:
      cnt2 += 1
      oov_set.add(word)
      continue
    embedding = full_vec[word]
    if word not in wordset:
      wordset.add(word)
      words.append(word)
      vecs.append(embedding)
  voc.close()
  print ("number of total words in query: "+str(cnt1))
  print ("number of total words oov: "+str(cnt2))
  print ("number of unique words in query: "+str(len(words)))
  print ("number of unique words oov: "+str(len(oov_set)))
  embedding_file = open(vocfile+'_pca_embed.txt','w')
  for i in range(len(words)):
    embedding_file.write(words[i]+' '+str(vecs[i][0])+' '+str(vecs[i][1])+'\n')
  embedding_file.close()
  hull = ConvexHull()
  for i in range(len(words)):
    hull.add(Point(vecs[i][0],vecs[i][1]),words[i])
  print("number of points on hull: ", str(len(hull.get_hull_points())))
  hull.display()

#visualize word embeddings
def plot_wordembedding_with_pca(vocfile):
  words = []
  vecs = []
  voc = open(vocfile,'r')
  for line in voc:
    word,embedding = line.strip().split(' ',1)  
    embedding = [float(e) for e in embedding.split()]
    if word and len(embedding)==100:
      words.append(word)
      vecs.append(embedding)
  voc.close()
  U,s,Vh = np.linalg.svd(vecs, full_matrices=False)
  colors = ['red','green','blue','yellow','orange','black']
  embedding_file = open(vocfile+'_pca_embed.txt','w')
  for i in range(len(U)):
    embedding_file.write(words[i]+' '+str(U[i,0])+' '+str(U[i,1])+'\n')
  embedding_file.close()
  print ("pca done")
  hull = ConvexHull()
  #fig = plt.gcf()
  #fig.set_size_inches(18.5,10.5)
  #plt.text(U[i,0],U[i,1],words[i],color=colori)
  #plt.plot(U[:,0],U[:,1],'o')
  '''
  for simplex in hull.simplices:
    plt.plot(U[simplex,0],U[simplex,1],'k-')
  plt.xlim((-0.5,0.5))
  plt.ylim((-0.5,0.5))
  plt.savefig('convex_hull_dec1_lda2.jpg')
  '''
  for i in range(len(U)):
    hull.add(Point(U[i,0],U[i,1]),words[i])
  print("number of points on hull: ", str(len(hull.get_hull_points())))
  hull.display()



def main(args):
  plot_wordembedding(args.word_vectors_voc_file) 

if __name__ == "__main__":
  # argument parse
  parser = argparse.ArgumentParser()
  parser.add_argument("--word_vectors_voc_file", help = "file of look up table of word embeddings and dec/lda keywords",required=True) 
  args = parser.parse_args()
  main(args)
