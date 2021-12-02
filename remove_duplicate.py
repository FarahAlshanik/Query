full_vec = open('/scratch2/yuhengd/boston/preprocessed/raw_tweet_embedding.voc','r')
out_vec = open('/scratch2/yuhengd/boston/preprocessed/raw_tweet_embedding_unique.voc','w')
wordset=set()
for line in full_vec:
  word = line.strip().split(' ')[0]
  if word not in wordset:
    wordset.add(word)
    out_vec.write(line)
out_vec.close()
full_vec.close()
