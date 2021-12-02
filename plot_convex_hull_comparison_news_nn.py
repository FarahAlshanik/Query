import argparse
import os
import numpy as np
import pylab
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
  def __init__(self):
    self._points = []
    self._hull_points = []

  def add(self, point):
    self._points.append(point)
    
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

    plt.title('Convex Hull')
    #plt.show()
    plt.savefig('convex_hull_lda3_v2.jpg')


def distance(word_vec,word1,word2):
  emd1,emd2 = word_vec[word1],word_vec[word2]
  return sum([(a-b)**2 for a,b in zip(emd1,emd2)])

def read_wordembedding(vocfile):
  word_vectors = []
  voc = open(vocfile,'r')
  for line in voc:
    word,embedding = line.strip().split(' ',1)  
    embedding = [float(e) for e in embedding.split()]
    word_vectors.append(embedding)
  voc.close()
  return word_vectors

#visualize word embeddings
def plot_wordembedding():
  words = []
  vecs = []
  colors = ['red','green','blue','yellow','orange','black']
  news_nn3_voc = read_wordembedding('news_nn3_query_result_embedding.voc_pca_embed.txt')
  lda3_voc = read_wordembedding('lda3_query_result_embedding.voc_pca_embed.txt')
  print('num points in lda voc '+str(len(lda3_voc)))
  print('num points in nn voc '+str(len(news_nn3_voc)))
  print ("read in pca embeddings done")
  news_nn3_hull = ConvexHull()
  lda3_hull = ConvexHull()
  for i in range(len(news_nn3_voc)):
    news_nn3_hull.add(Point(news_nn3_voc[i][0],news_nn3_voc[i][1]))
  for i in range(len(lda3_voc)):
    lda3_hull.add(Point(lda3_voc[i][0],lda3_voc[i][1]))
  news_nn3_hull_points = news_nn3_hull.get_hull_points()
  lda3_hull_points = lda3_hull.get_hull_points()
  print("number of points on news_nn3_hull: ", str(len(news_nn3_hull_points)))
  print("number of points on lda3_hull: ", str(len(lda3_hull_points)))
  #plot 2 convex hulls
  # hull points
  hx = [p.x for p in news_nn3_hull_points]
  hy = [p.y for p in news_nn3_hull_points]
  plt.plot(hx, hy,color='red',label='news_nn3')
  hx = [p.x for p in lda3_hull_points]
  hy = [p.y for p in lda3_hull_points]
  plt.plot(hx, hy,color='blue',label='lda3')

  plt.title('Convex Hull of Query')
  #plt.show()
  plt.legend()
  plt.savefig('convex_hull_lda3_newsnn3.jpg')


def main(args):
  plot_wordembedding() 

if __name__ == "__main__":
  # argument parse
  parser = argparse.ArgumentParser()
  #parser.add_argument("--word_vectors_voc_file", help = "file of look up table of word embeddings and dec/lda keywords",required=True) 
  args = parser.parse_args()
  main(args)
