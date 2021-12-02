import argparse
from os import listdir
from os.path import isfile, join
from itertools import islice
import sys

NUM_TOP_DEC_WORDS = 200

def get_files(file_folder='', file_format='ecentrality%d.txt'):
	"""
	Return all interval files in a given folder.
	Usage:
		files = get_files('/zfs/dicelab/farah/Baltimore/dec_vals_new2/', 'file_%d.csv')', 'file_%d.csv')
		# files = ['hourly_intervals/file_1.csv' ... 'hourly_inverals/file_216.csv']

	:param file_folder: where your dec input files are
	:param file_format: how your files are named
	:returns files: ordered list of files to process.
	"""
	if len(file_folder):
		if file_folder[-1] != '/':
			file_folder += '/'

	num_files = len([f for f in listdir(file_folder) if isfile(join(file_folder, f))])

	files = []
	for i in range(0, num_files):
		files.append(file_folder + file_format % (i + 1))
		print(files[-1])

	return files


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

    for interval, f in enumerate(input_files):
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
        if jaccard >= 0.35:
            emerging_windows.append(interval+1)

    return emerging_windows


def build_args():
    """
    Build out program arguments and return them.
    :returns: arguments pased by the user.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder",\
                help="Location of the folder where your (numbered/ordered) interval files are.",\
                type=str,\
                default='/zfs/dicelab/farah/Baltimore/dec_vals_new2/')

    parser.add_argument("--output_file", help="output file for emerging windows", type=str, default='/zfs/dicelab/farah/Baltimore/emerging_windows_stop.txt')

    parser.add_argument("--log_file", help="File to write output to.", type=str, default=None)

    # parser.add_argument("-v", "--verbosity", help="Prints various log messages", type=bool)

    return parser.parse_args()

if __name__ == "__main__":
    # argument parsing and some small format-checking
	args = build_args()

	if args.log_file:
		sys.stdout = open(args.log_file, 'w')


	if len(args.input_folder):
		if args.input_folder[-1] != '/':
			args.input_folder += '/'
			print(args.input_folder)

	input_files = get_files(file_folder=args.input_folder)
	emerging_windows = calculate_jaccard(3, input_files)
	with open(args.output_file,'w') as outputf:
		for w in emerging_windows:
			outputf.write(str(w)+'\n')
    

