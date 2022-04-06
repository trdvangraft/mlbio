import pandas as pd
import numpy as np
import math
import random
from numba import jit


def generateNoise(level, concentration):
    data = pd.read_csv("../data/Lindel_training.txt", sep='\t', header=None)
    editData = data.copy()

    x_t = editData.iloc[:, 1:3034]  # the full one hot encoding
    # 557 observed outcome frequencies
    y_t = editData.iloc[:, 3034:]

    def normalization(labeli, noise):
        return labeli * (1 - level) + noise * level

    for index, row in editData.iterrows():
        numberofzeros = 557 - math.ceil(concentration * 557)
        zeros = np.zeros(numberofzeros)
        randomNoises = np.random.uniform(0.00, 1.00, math.ceil(concentration * 557))
        total = np.concatenate([randomNoises, zeros])
        np.random.shuffle(total)
        normalizedRandomNoise = total / total.sum()
        for j in range(0, 557):
            editData.at[index, 3034 + j] = (normalization(row[3034 + j], normalizedRandomNoise[j]))

    filename = "./Lindel_training_withnoise_" + str(int(level * 100)) + "_" + str(int(concentration * 100)) + ".txt"
    editData.to_csv(filename, sep='\t', index=False, header=None)
    print("---- random generator finished for level: " + str(level) + " ----")


def generateBootstrappingNoise(samples, repetitions):
    # averageReadsperClass = 1.16e6 / 6872
    # samples = math.ceil(percentage * averageReadsperClass)
    data = pd.read_csv("../data/Lindel_training.txt", sep='\t', header=None)
    editData = data.copy().to_numpy()

    y_t = editData[:, 3034:]
    labelSize = y_t.shape[1]

    for rep in range(repetitions):
        for row in editData:
            bootstrappedList = np.zeros(labelSize)
            rangeRow = list(range(labelSize))
            sampledRow = random.choices(rangeRow,weights=row[3034:],k=samples)
            for label in sampledRow:
                bootstrappedList[label] += 1

            bootstrappedList = bootstrappedList / sum(bootstrappedList)

            row[3034:] = bootstrappedList

        resultData = pd.DataFrame(editData)

        filename = "./Lindel_training_bootstrapping_" + str(int(samples)) + "samples_" + str(rep) + ".txt"
        resultData.to_csv(filename, sep='\t', index=False, header=None)
        print("---- random generator finished for: " + str(samples) + " samples, repetition: " + str(rep) + " ----")


if __name__ == "__main__":
    percentages = [10,20]
    for i in percentages:
        generateBootstrappingNoise(i, 5)
