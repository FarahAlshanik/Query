import csv
processed_files=[]
with open("/zfs/dicelab/farah/query_exp/query-construction2/procc_files.csv",'r',encoding='utf-8') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        processed_files.append(row)

print(len(processed_files))
print(len(processed_files[0]))






#print(processed_files[0])


w=5

def counter(t):
    mm=[]
    documents=t
    #print(t)
    r=0

    for o in range(len(documents)):
        for u in documents[r]:
            mm.append(u)
        r+=1
    #print(mm)
    return mm




k=0
wind=[]

count=0
for i in range(len(processed_files)):
    if(i<w-1):
        #print(times[i], i)
        wind.append(processed_files[i])



        k+=1
    elif(i==w-1):
        #print(times[0:w],s[0:w])
        wind.append(counter(processed_files[0:w]))
        k+=1
        count+=1
    elif(i>w-1):


        k+=1
        #print(times[count:w+1], s[count:w+1])
        wind.append(counter(processed_files[count:w+1]))
        w+=1
        count+=1


print(wind[0])



for i in range(len(wind)):
	d=open("/zfs/dicelab/farah/query_exp/query-construction2/sereen/"+str(i)+".txt",'w')

	for i in wind[i]:
    		d.write('%s ' % i)
    		d.write('\n')


	d.close()
