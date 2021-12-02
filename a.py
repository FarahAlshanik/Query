from collections import Counter

hotwords = ('tweet', 'twitter')

lines = "abc abc suj suj"
cc=lines.split()
c = Counter(lines.split())

for hotword in set(cc):
    print (hotword, c[hotword])
