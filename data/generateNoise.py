import pandas as pd
import numpy as np

def generateNoise(level):
    data = pd.read_csv("./Lindel_training.txt", sep='\t', header=None)
    editData = data.copy()

    x_t = editData.iloc[:, 1:3034]  # the full one hot encoding
    # 557 observed outcome frequencies
    y_t = editData.iloc[:, 3034:]

    randomNoises = np.random.uniform(0.000000,1.000000, 557)
    normalizedRandomNoise = randomNoises/randomNoises.sum()

    def normalization(labeli, noise):
        return labeli * (1-level) + noise * level

    for index, row in editData.iterrows():
        for j in range(0,557):
            editData.at[index,3034+j] = (normalization(row[3034+j],normalizedRandomNoise[j]))


    filename = "./Lindel_training_withnoise_" + str(int(level*100)) + ".txt"
    editData.to_csv(filename, sep='\t', index=False, header=None)
    print("random generator finished for level: " + str(level))


if __name__ == "__main__":
    generateNoise(0.1)
    generateNoise(0.2)
    generateNoise(0.3)
    generateNoise(0.4)
    generateNoise(0.5)
