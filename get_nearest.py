import argparse
import os
import numpy as np
from os import listdir
from os.path import isfile, join
from pyfasttext import FastText
#import higest_freq_param2
import pandas as pd
import re


model = FastText('/zfs/dicelab/farah/model_skip_300.bin')
word_nn=model.nearest_neighbors('protest',k=20)
print(word_nn)

