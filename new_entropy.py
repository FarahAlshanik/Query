#cnt_times = collections.Counter()
dict = {}
k=0
len_files=[]# to have the number of documents in each 2 time interval
s=0
#count_times=[]
for i in range(len(times)):
    #s=cnt_times
    #count_times.append(s)
    dict[idf_file[k]]=collections.Counter()
    #cnt_times_new = collections.Counter()
    if(k==0):
        s+=len(times[i])
    for doc in (times[i]):
         dict[idf_file[k]][doc]+=1
    if(k!=0):
        dict[idf_file[k]]=dict[idf_file[k]]+dict[idf_file[k-1]]
        s+=len(times[i])
        len_files.append(s) 
    else:
        len_files.append(len(times[i]))
                

        
    #print(i)
    #print(cnt_times)
    k+=1
   
    #print(s)
    #for k,v in  cnt_times.most_common():
     #   f.write(k+":"+str(v)+" " )
    #f.write("\n")
    
    
