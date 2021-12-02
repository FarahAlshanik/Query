def read_wordembedding(vocfile,word_vectors):
  voc = open(vocfile,'r')
  for line in voc:
    word,embedding = line.strip().split(' ',1)
    embedding = [float(e) for e in embedding.split()]
    word_vectors[word]=embedding
  voc.close()

word_vec = {}
read_wordembdding('/zfs/dicelab/farah/query_exp/query-construction2/Boston/model.vec')
