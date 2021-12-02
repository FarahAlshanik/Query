import time
import datetime
import csv
csvfile2 = open("/zfs/dicelab/farah/query_exp/query-construction2/Baltimore/curfew.csv",'w')
f1 = csv.writer(csvfile2, delimiter=',')
i=0
with open('/zfs/dicelab/farah/query_exp/query-construction2/Baltimore/sorted.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            tId = line[2]; #tweet_long.partition(',')[2]
            tweetText = line[1]
            tDate = line[0]
            i+=1
            print(i)
            if('curfew' in tweetText.lower().split()):
                f1.writerow(line)
