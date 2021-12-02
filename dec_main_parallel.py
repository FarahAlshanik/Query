"""
Main implementation of "https://people.cs.clemson.edu/~isafro/papers/dynamic-centralities.pdf"

Grace Glenn (mgglenn@g.clemson.edu)
November 2017
"""
import argparse
import networkx as nx
from os import listdir
from os.path import isfile, join
import sys
from itertools import islice

# custom imports
import dec_text
import dec_graph
import shutil



NUM_TOP_DEC_WORDS = 200

def get_files(file_folder='', file_format='file_%d.csv'):
	"""
	Return all interval files in a given folder.
	Usage:
		#files = get_files('/zfs/dicelab/farah/query_exp/query-construction2/Baltimore/interval/int/', 'file_%d.csv')
                files=get_files('/zfs/dicelab/farah/query_exp/query-construction2/Baltimore/interval/queryoutput/lda_int15/', 'file_%d.csv')
		# files = ['hourly_intervals/file_1.csv' ... 'hourly_inverals/file_216.csv']

	:param file_folder: where your files are
	:param file_format: how your files are named
	:returns files: ordered list of files to process.
	"""
	if len(file_folder):
		if file_folder[-1] != '/':
			file_folder += '/'
	
	num_files = len([f for f in listdir(file_folder) if isfile(join(file_folder, f))])

	files=[]
	s=""
	topic0=[]
	topic1=[]
	topic2=[]
	topic3=[]
	topic4=[]
	for i in sorted(listdir(file_folder)):
		#print(i)
		if(i[5:6]=='0'):
			topic0.append(file_folder+i)
		if(i[5:6]=='1'):
			topic1.append(file_folder+i)
		if(i[5:6]=='2'):
                	topic2.append(file_folder+i)
		if(i[5:6]=='3'):
                	topic3.append(file_folder+i)
		if(i[5:6]=='4'):
			topic4.append(file_folder+i)
	#	files.append(file_folder+i)

		s=i
	#print(files[0]) #files has the full path
#	print(s[5:6])#to return topic number
#	print(len(topic0))
#	print(len(topic1))
#	print(len(topic2))
#	print(len(topic3))
#	print(len(files))
	
	files.append(topic0)
	files.append(topic1)
	files.append(topic2)
	files.append(topic3)
	files.append(topic4)
#	print(files[0])
	return files


def build_args():
	"""
	Build out program arguments and return them.
	:returns: arguments pased by the user.
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument("--input_folder",\
			    help="Location of the folder where your (numbered/ordered) interval files are.",\
			    type=str,\
			    default="/zfs/dicelab/farah/query_exp/query-construction2/Baltimore/interval/queryoutput/lda_int15/")

	parser.add_argument("--output_folder", help="location of output files", type=str, default='/zfs/dicelab/farah/query_exp/query-construction2/Baltimore/step1_lda/')

	parser.add_argument("--P", help="Number of intervals to calculate DEC from.", type=int, default=5)

	parser.add_argument("--log_file", help="File to write output to.", type=str, default=None)
        
	# parser.add_argument("-v", "--verbosity", help="Prints various log messages", type=bool)

	return parser.parse_args()

def jaccard_similarity(list1, list2):
    """

    :param list1: first list of top dec words
    :param list2: second list for top dec words
    :return jaccard: return the jaccard similarity of the 2 lists of words
    """
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

def get_top_dec_words(inputfile):
    """

    :param inputfile: dec files
    :return list: list of top 200 dec words in a list
    """
    with open(inputfile) as inputf:
        head = list(islice(inputf, NUM_TOP_DEC_WORDS))
        headlist = [ w.strip().split()[0] for w in head]
        return headlist


def calculate_jaccard(P=3, input_files=[]):

    """
        Calculate jaccard value of top 200 words between current window and 3-prev window.

        :param P: window of how many past ec values to use in comparing jaccard similarity
        :param files: (ORDERED) list of files to process, contains sorted dec words
        """
    emerging_windows = []
    files=[]
    for interval, f in enumerate(input_files):
        r=f.find(".")
        print(f[100:r])
        print(f)
        
        if interval < P :
            # skip the first P(3) windwows
            continue
        print("Processing data from interval %d" % (interval + 1))

        # read in top dec words from the 2 windows
        list1 = get_top_dec_words(f)
        prev_f = input_files[interval-P]
        list2 = get_top_dec_words(prev_f)
        jaccard = jaccard_similarity(list1,list2)

        # add recognized interval to emerging_windows
        if jaccard <= 0.15:
            emerging_windows.append(interval+1)
            files.append("topic0_file_"+f[100:r]+".csv"+ " "+str(interval+1))
    #return emerging_windows
    return files



def calculate_DEC(P=5, input_files=[], output_folder=''):
	"""
	Calculate DEC values on all text files.

	:param P: window of how many past ec values to use in slope (creating dec)
	:param files: (ORDERED) list of files to process, contains data
	"""
	#G = nx.Graph()
	#ec_windows = {}
	#buckets = dec_graph.initialize_buckets(P=P)
	stopwords = dec_text.getStopwords()
	topic0=[]
	topic1=[]
	topic2=[]
	topic3=[]
	topic4=[]
	files=[]
	for i in range(len(input_files)):
		G = nx.Graph()
		ec_windows = {}
		buckets = dec_graph.initialize_buckets(P=P)
		for interval, f in enumerate(input_files[i]):
			print("Processing data from interval %d" % (interval + 1))

		# decrease edges, remove zero-weight edges and zero-degree nodes
			bucket_index = interval % P
			deleted = dec_graph.decrease_edge_weights_update_graph(G=G,\
					current_bucket=buckets[bucket_index],\
							ec_windows=ec_windows)

		# read in text from interval, update edge weights and node degrees 
			text = dec_text.get_text_from_file(file=f, stopwords=stopwords)
			dec_graph.update_graph_with_text(G=G,\
					    text=text,\
					    current_bucket=buckets[bucket_index])

			print("\tNum Keywords: %d" % len(G))

		# calculate DEC values and write them to file
			dec_vals = dec_graph.compute_dec_vals(G=G, ec_windows=ec_windows, P=P)

			outfile = output_folder + 'ecentrality_topic'+str(i)+ "_"+f[90:]+".txt"
			print(f[90:])
			if(i==0):
				topic0.append(outfile)
			if(i==1):
				topic1.append(outfile)
			if(i==2):
				topic2.append(outfile)
			if(i==3):
				topic3.append(outfile)
			if(i==4):
				topic4.append(outfile)
		
			
			dec_text.write_dec_values(outfile=outfile, dec_vals=dec_vals, rank=True)
#	print(topic[0])
	files.append(topic0)
	files.append(topic1)
	files.append(topic2)
	files.append(topic3)
	files.append(topic4)
	input_folder="/zfs/dicelab/farah/query_exp/query-construction2/Baltimore/interval/queryoutput/lda_int15/"
	for i in range(len(files)):
		emerging_windows = calculate_jaccard(3, files[i])
		with open(output_folder+"emerging_windows_topic_"+str(i) +".txt",'w') as outputf:
			for w in emerging_windows:
				outputf.write(str(w)+'\n')
				shutil.copy(input_folder+str(w).split()[0], output_folder+'/topic'+str(i))
	#for i in range(len(files)):
	#	for f in files[i]:
    	#		shutil.copy(f, output_folder+'/topic'+str(i))


if __name__ == "__main__":
	# argument parsing and some small format-checking
	args = build_args()

	if args.log_file:
	    sys.stdout = open(args.log_file, 'w')

	if len(args.output_folder):
		if args.output_folder[-1] != '/':
			args.output_folder += '/'
			print(args.output_folder)

	if len(args.input_folder):
		if args.input_folder[-1] != '/':
			args.input_folder += '/'
			print(args.input_folder)

	input_files = get_files(file_folder=args.input_folder)
	print(input_files[0])
	calculate_DEC(input_files=input_files, P=args.P, output_folder=args.output_folder)
