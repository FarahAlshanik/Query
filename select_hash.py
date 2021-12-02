import pandas as pd

hashtags = dict()
ff=pd.read_csv('/zfs/dicelab/farah/balt_break_all_tweet/file_155.csv',names=['a','tweet','c'])

for tweet in ff['tweet']:
    content =  tweet
    for word in content.strip().split():
        if word.lower().startswith("#"):
            hashtags[word.lower()] = hashtags.get(word.lower(),0)+1
                
sorted_hashtags = sorted(hashtags.items(), key = lambda kv: kv[1],reverse=True)
        
for k,v in sorted_hashtags:
    print(k+" "+str(v)+"\n")

df = pd.DataFrame(sorted_hashtags,columns =['hash','count']) 

print(df)
for i,j in df[['hash','count']].itertuples(index=False):  
    print(i,j)


print(len(df))
