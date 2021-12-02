i=0
phr=''
pday=''
pmin=0
interval_len = 15
interval = -1

    
with open('/zfs/dicelab/farah/query_exp/query-construction2/Baltimore/interval/queryoutput/lda_int15/t0', 'rt') as csvfile:
    reader = csv.reader(csvfile)
       
    for line in reader:
       # print(line)
        tId = line[2]; 
        tweetText = line[1]
        tDate = line[0]
        tweetDate = datetime.datetime.strptime(tDate, "%Y-%m-%d %H:%M:%S-04:00")
        day=tweetDate.day
        hr=tweetDate.hour
        minute=tweetDate.minute
        curr_interval = minute / interval_len

        break_file = False
        if interval_len == 60:
            
            break_file = (i is 0 or hr is not phr)
        else:
            break_file = (i is 0 or curr_interval != interval)

        if break_file:
            interval = curr_interval
            pmin = minute
            phr = hr
            
            if(minute==0):
                print(str(tweetDate))
                i = i+1
                csvfile2 = open('/zfs/dicelab/farah/query_exp/query-construction2/pp' + "/file_" + str(i)+".csv",'wt')
                f1 = csv.writer(csvfile2, delimiter=',')
            if(minute==16):
                print(str(tweetDate))
                i = i+1
                csvfile2 = open('/zfs/dicelab/farah/query_exp/query-construction2/pp' + "/file_" + str(i)+".csv",'wt')
                f1 = csv.writer(csvfile2, delimiter=',')
            if(minute==31):
                print(str(tweetDate))
                i = i+1
                csvfile2 = open('/zfs/dicelab/farah/query_exp/query-construction2/pp' + "/file_" + str(i)+".csv",'wt')
                f1 = csv.writer(csvfile2, delimiter=',')
            if(minute==46):
                print(str(tweetDate))
                i = i+1
                csvfile2 = open('/zfs/dicelab/farah/query_exp/query-construction2/pp' + "/file_" + str(i)+".csv",'wt')
                f1 = csv.writer(csvfile2, delimiter=',')
            if(minute==60):
                print(str(tweetDate))
                i = i+1
                csvfile2 = open('/zfs/dicelab/farah/query_exp/query-construction2/pp' + "/file_" + str(i)+".csv",'wt')
                f1 = csv.writer(csvfile2, delimiter=',')
            
            
        f1.writerow(line)

