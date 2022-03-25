import pandas as pd
import numpy as np
import math
import random

def generateNoise(level, concentration):
    data = pd.read_csv("./Lindel_training.txt", sep='\t', header=None)
    editData = data.copy()

    x_t = editData.iloc[:, 1:3034]  # the full one hot encoding
    # 557 observed outcome frequencies
    y_t = editData.iloc[:, 3034:]

    def normalization(labeli, noise):
        return labeli * (1-level) + noise * level

    for index, row in editData.iterrows():
        numberofzeros = 557 - math.ceil(concentration*557)
        zeros = np.zeros(numberofzeros)
        randomNoises = np.random.uniform(0.00, 1.00, math.ceil(concentration*557))
        total = np.concatenate([randomNoises, zeros])
        np.random.shuffle(total)
        normalizedRandomNoise = total/total.sum()
        for j in range(0,557):
            editData.at[index,3034+j] = (normalization(row[3034+j],normalizedRandomNoise[j]))


    filename = "./Lindel_training_withnoise_" + str(int(level*100)) + "_" + str(int(concentration*100)) + ".txt"
    editData.to_csv(filename, sep='\t', index=False, header=None)
    print("---- random generator finished for level: " + str(level) + " ----")


if __name__ == "__main__":
    levels = [0.1,0.2, 0.3, 0.4, 0.5]
    concentrations = [1, 0.1, 0.01]
    for l in levels:
        for c in concentrations:
            generateNoise(l, c)
