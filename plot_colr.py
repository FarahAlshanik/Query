import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pylab
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import matplotlib.pyplot as plt
import random

# simulate data
# =============================
'''
xx=[]
yy=[]
for i in range(300):
	xx.append(i)
	yy.append(i*5)


label=['one','two','three']

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

x=np.asarray(xx)
y=np.asarray(yy)

print(xx)

#dydx = np.cos(96 * (x[:-1] + x[1:]))  # first derivative

dydx=x

print(dydx)

#print(len(dydx))


# Create a set of line segments so that we can color them individually
# This creates the points as a N x 1 x 2 array so that we can stack points
# together easily to get the segments. The segments array for line collection
# needs to be (numlines) x (points per line) x 2 (for x and y)

points = np.array([x, y]).T.reshape(-1, 1, 2)

#points=np.array([x, y])
segments = np.concatenate([points[:-1], points[1:]], axis=1)

fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)

# Create a continuous norm to map from data points to colors

norm = plt.Normalize(dydx.min(), dydx.max())

lc = LineCollection(segments, cmap='viridis', norm=norm)
# Set the values used for colormapping

lc.set_array(dydx)
lc.set_linewidth(2)

line = axs[0].add_collection(lc)

fig.colorbar(line, ax=axs[0])

# Use a boundary norm instead
cmap = ListedColormap(['r', 'g', 'b'])
norm = BoundaryNorm([1, 96, 193, 289], cmap.N)

lc = LineCollection(segments, cmap=cmap, norm=norm)

lc.set_array(dydx)

lc.set_linewidth(2)

line = axs[1].add_collection(lc)

fig.colorbar(line, ax=axs[1])

axs[0].set_xlim(x.min(), x.max())

#axs[0].set_ylim(-1.1, 1.1)

#plt.show()

plt.savefig('c.png')



import numpy as np
import pylab as pl
from matplotlib import collections  as mc

segments = []
colors = np.zeros(shape=(300,4))

i = 0

z=x
k=0
for i in range(len(z)):
	if(k==96):
		z[i]=96
		k=0
	k+=1	

z=np.asarray(z)

print(z)

for x1, y1, z1 in zip(x, y,z):

    if z1==96:
        colors[i] = tuple([1,0,0,1])
    elif z1 !=96:
        colors[i] = tuple([0,1,0,1])
    segments.append([(x1),(y1)])
    if(i==299):
        break
    i += 1
for x1, x2, y1,y2, z1,z2 in zip(x, x[1:], y, y[1:],z, z[1:]):
    if z1==96:
        colors[i] = tuple([1,0,0,1])
    elif z1 !=96:
        colors[i] = tuple([0,1,0,1])
    #else:
       # colors[i] = tuple([0,0,1,1])
    segments.append([(x1, y1), (x2, y2)])
    if(i==299):
        break
    i += 1     


lc = mc.LineCollection(segments, colors=colors, linewidths=2)
fig, ax = pl.subplots()
ax.add_collection(lc)
ax.autoscale()
ax.margins(0.1)
#pl.show()
'''
 
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
        l = plt.gca().plot(x[start:end],y[start:end],lw=2,c=seg[0]) 
        start = end
'''

x = np.arange(1000) 
y = np.random.randn(1000) # randomly generated values

# color segments
colors = ['blue']*1000
colors[300:500] = ['red']*200
colors[800:900] = ['green']*100
colors[600:700] = ['magenta']*100
'''
plot_multicolored_lines(x,y,colors)
#plt.show()
plt.savefig('c.png')
'''
