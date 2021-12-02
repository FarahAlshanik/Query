w=5
k=0
cnt=[]
s=[0,1,2,3,4,5,6,7,8,9]
count=0
data=[]
for i in range(len(times)):
    
    if(i<w-1):
        print(times[i])
        #print(times[i], i)
        #data.append(counter_lesswindow(times[i]))
         
         
        k+=1
    elif(i==w-1):
        print(times[0:w])
        data.append(times[0:w])
        #cnt.append(counter(times[0:w],0,w))
        k+=1
        count+=1
    elif(i>w-1):
        k+=1
        if(len(times[count:w+1])==5):
            print(times[count:w+1])
            data.append(times[count:w+1])
        #cnt.append(counter(times[count:w+1],count,w+1))
        w+=1
        count+=1


#######################For tweet Entropy
def idf(data):
    idf_file_data=[]
    s=[]
    for i in data:
        for j in i:
            s.append(j)
    #print(s)
    
    for (IDF, term) in calculate_idf(s):
        idf_file_data.append([IDF, term])
    return idf_file_data

idf_data=[]
for i in data:
    idf_data.append(idf(i))
    


def idf_dic(t):
    idf_dict={}
    for i in t:
        idf_dict[i[1]]=i[0]
    return idf_dict
    

idf_data_dict=[]
for i in range(len(idf_data)):
    idf_data_dict.append(idf_dic(idf_data[i]))

 
from gensim.models import Word2Vec
model_all=Word2Vec(times, size=100, window=5, min_count=0, workers=4)


print("Done Combine corpus for word2vec")

ee_all=[]
for i, word in enumerate(model_all.wv.vocab):

    ee_all.append([word,model_all[word]])

#ee_file1_2
emb_all={}
for i in ee_all:
    #print(i[0])
    emb_all[i[0]]=i[1]


def get_avg(data,k):
    s=[]
    for i in data:
        for j in i:
            s.append(j)
    avg_file1=[]
    for i in s:
        avg_file1.append(get_centroid_idf(i,emb_all,idf_data_dict[k],100))
        
    final_cent_array_file1=[] # now I got the word centroid for each doc 
    for i in avg_file1:
        final_cent_array_file1.append( np.array(i, dtype=np.float32).reshape(( 100)))
        
    return final_cent_array_file1



get_average=[]
for j,i in enumerate(data):
    #print(i)
    get_average.append(get_avg(i,j)) 
    
 # we need to find average for all

avg=[]
for i in range (len(get_average)):
    for j in get_average[i]:
        ss+=i
    avg.append(ss/len(get_average[i]))

###############TO do similarity
from scipy import spatial
for i in range(len(avg)):
    if(i==len(avg)-1):
        break
    print(1-spatial.distance.cosine(avg[i],avg[i+1]))
    #print(data[i+1])
    print("***************************")






