import pandas as pd
import math
import numpy as np
import random

data = pd.read_csv("Lindel_training.txt", sep='\t', header=None)
datanoised = pd.read_csv("Lindel_training_withnoise_50_100.txt", sep='\t', header=None)

print(data.iloc[0, 3034:])
print(datanoised.iloc[0, 3034:])
print(datanoised.iloc[0,3034:].sum())
print("stop")



