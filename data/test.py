import numpy as np

randomNoises = np.random.uniform(0.000000,1.000000, 500)
normalisedNoise = randomNoises/randomNoises.sum()

print(randomNoises)
print(normalisedNoise)
print(normalisedNoise.sum())
