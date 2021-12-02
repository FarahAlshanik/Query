import pandas as pd
f=pd.read_csv('/zfs/dicelab/farah/query_exp/query-construction2/Baltimore/sorted.csv',names=['a','b','c'])
print(f.head())
from nltk.tokenize import word_tokenize

tweetText = f['b'].apply(word_tokenize)
tweetText.head()

tokens_baltimore=[]
for i in tweetText:
    for j in i:
        tokens_baltimore.append(j.lower())

df = pd.DataFrame (tokens_baltimore,columns=['tokens'])

df.to_csv('/zfs/dicelab/farah/dec_results/balt_tokens2.csv',index=False)

