 
import pylab
import matplotlib.pyplot as plt
plt.switch_backend('agg')
 
import matplotlib.pyplot as plt  
import random

 
s=[]

with open('sw96t1a.txt' , 'r',newline='') as f:
    x = f.read().splitlines()
    s=[]
    for i in (x):
        s.append(i)



#print(s[1].split(' ')[0])

x_data=[]
y_data=[]



for i in s:
    y_data.append(float(i.split(' ')[0]))
    x_data.append(int(i.split(' ')[1])+96)

print(x_data)

for i in range(1536):
    if(i not in x_data):
        print(str(i)+",")

#print(y_data)
plt.plot(x_data,y_data)
#plt.figure(figsize=(8,8))
plt.xlabel('Time')
plt.ylabel('Average Pairwise Jaccard Similarity')

#plt.ylabel('Jaccard Entropy')

plt.savefig('t196_new.png')



'''
ss=[]
for i in :
	ss.append(i)

jj=ss

for i in jj:
	print(i)

print(len(jj))
print(jj[0])

f_n=[]
with open('files_name_entropy', 'r') as f:
    f_n.append( f.read())
    #dic = ast.literal_eval(s)


file_name=[]
for i in f_n[0].split("\n"):
    file_name.append(i)


file_nmubers=[]
for i in file_name:
    f=i.split(".")[0]
    h=f[5:]
    if(h!=''):
    	file_nmubers.append(int(f[5:]))
    
print(len(file_nmubers))

entropy_file_num=[]
for i in range(len(jj)):
    entropy_file_num.append([jj[i],file_nmubers[i]])

ss=sorted(entropy_file_num,key = lambda x: x[0])
for i in ss:
	print(i)
#print(ss)

with open('hig_low_entropy.txt', 'w') as filehandle:
    for listitem in ss:
   

     filehandle.write('%s\n' % listitem)


'''

'''

x_data=[]
y_data=[]
j=0
for i in ss:
    y_data.append(i[0])
    x_data.append(i[1])
    #j+=1


 

 

'''
 
# From here the plotting starts
'''
plt.scatter(x_data, y_data, c='r', label='data')
#plt.plot(x_func, y_func, label='$f(x) = 0.388 x^2$')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Entropy')
plt.legend()
plt.show()
plt.savefig('word_entropy.jpg')
'''




#plt.plot(x_data,y_data)
#plt.savefig('word_entropy_line_new.jpg')

