import argparse
from os import listdir
import argparse
from os import listdir
import os
#parser = argparse.ArgumentParser()
#args = parser.parse_args()
#input_files = os.listdir(args.input_dir)
#input_files =  get_files(file_folder="/zfs/dicelab/farah/break_police_15M")

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir",help="input directory of query result files to count hashtags", required=True)
args = parser.parse_args()
input_files = os.listdir(args.input_dir)

#input_files =  get_files(file_folder="/zfs/dicelab/farah/query_exp/query-construction2/sample")

#input_dirc='/zfs/dicelab/farah/query_exp/query-construction2/sample/'
#input_dirc='/zfs/dicelab/farah/break_police_15M/'

input_dirc='/zfs/dicelab/farah/query_exp/query-construction2//Baltimore/interval/queryoutput/lda_int15/t0/' 


split_numbers=[]


for i in input_files:
    split_numbers.append(int(i.split('_')[2].split('.')[0]))
    #print(split_numbers)
    sorted_num=sorted(split_numbers)
    files=[]
    for i in sorted_num:
        files.append("file"+"_"+str(i)+".csv")


print(sorted_num)

full_path=[]
for interval, f in enumerate(files):
    print("Processing data from file " + f)
    inputf0 = os.path.join(f)
    full_path.append(input_dirc+f)
    #print(inputf0)
    #file=pd.read_csv(inputf0,encoding='latin-1',sep=',', names=["time", "abstract", "id"])

print(full_path[0])





