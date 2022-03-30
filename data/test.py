import pandas as pd
import math
import numpy as np
import random
from numba import jit
from itertools import groupby
import collections

ordata = pd.read_csv("./Lindel_training.txt", sep='\t', header=None)
data = pd.read_csv("./Lindel_training_bootstrapping_40_0.txt", sep='\t', header=None)
print(ordata.iloc[:5,3034:])
print(data.iloc[:5,3034:])
print("this")
