import matplotlib.pyplot as plt
import numpy as np

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

x=[]
y=[]
for i in range(1530):
	x.append(i)
	y.append(i*5)


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



 
plot_multicolored_lines(x,y,colors)
#plt.show()
plt.savefig('c.png')
