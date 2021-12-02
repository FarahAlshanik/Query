 
import pylab
import matplotlib.pyplot as plt
plt.switch_backend('agg')
 
import matplotlib.pyplot as plt  
import random


def find_contiguous_colors(colors):
    # finds the continuous segments of colors and returns those segments
    segs = []
    curr_seg = []
    prev_color = ''
    for c in colors:
        if c == prev_color or prev_color == '':
            curr_seg.append(c)
        else:
            segs.append(curr_seg)
            curr_seg = []
            curr_seg.append(c)
        prev_color = c
    segs.append(curr_seg) # the final one
    return segs

def plot_multicolored_lines(x,y,colors):
    segments = find_contiguous_colors(colors)
    plt.figure()
    start= 0
    for seg in segments:
        end = start + len(seg)
        l, = plt.gca().plot(x[start:end],y[start:end],lw=2,c=seg[0])
        start = end

#x = np.arange(1000)
#y = np.random.randn(1000) # randomly generated values
# color segments



 
s=[]

with open('so.txt' , 'r',newline='') as f:
    x = f.read().splitlines()
    s=[]
    for i in (x):
        s.append(i)



#print(s[1].split(' ')[0])

x_data=[]
y_data=[]
for i in s:
    y_data.append(float(i.split(' ')[0]))
    x_data.append(int(i.split(' ')[1]))



for i in range(1536):
    if(i not in x_data):
        print(str(i)+",")

#print(y_data)
plt.plot(x_data,y_data)
#plt.figure(figsize=(8,8))
#plt.savefig('t1.png')





colors = ['blue']*96
colors[97:192] = ['red']*96
colors[193:288] = ['green']*96
colors[289:385] = ['magenta']*96

colors[386:481] = ['blue']*100
colors[482:577] = ['red']*100
colors[578:673] = ['green']*100
colors[674:769] = ['magenta']*100

colors[770:865] = ['blue']*100
colors[866:961] = ['red']*100
colors[962:1057] = ['green']*100
colors[1058:1153] = ['magenta']*100


colors[1154:1249] = ['blue']*100
colors[1250:1345] = ['red']*100
colors[1346:1441] = ['green']*100
colors[1442:1530] = ['magenta']*100




plot_multicolored_lines(x_data,y_data,colors)
plt.xlabel('Time')
plt.ylabel('Jaccard Entropy')

#plt.show()
plt.savefig('soc.png')







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

