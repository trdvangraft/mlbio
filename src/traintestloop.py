'''
This file will execute the training and testing multiple times
'''

from src.mainTraining import main as maintraining
from visualizations.visMSE import generateOutcomes


def oneTestIteration(traingset, trainingtype):
    maintraining(traingset)
    generateOutcomes(trainingtype)


def uniformExperiments():
    '''
    Running training and testing for uniform noise experiments
    '''
    levels = [10, 20, 30, 40, 50]
    concentration = [100]
    lst = []
    typelst = []
    for l in levels:
        for c in concentration:
            lst.append(f"../data/Lindel_training_withnoise_{l}_{c}.txt")
            typelst.append(f"{l}noise_{c}concentration")

    for j in range(len(lst)):
        oneTestIteration(lst[j], typelst[j])
        print(f"---- {typelst[j]} finished ----")


def bootstrapExperiments():
    levels = [10, 20, 34, 68, 102, 136]
    repetitions = [0, 1, 2, 3, 4]
    lst = []
    typelst = []
    for l in levels:
        for r in repetitions:
            lst.append(f"../data/Lindel_training_bootstrapping_{l}samples_{r}.txt")
            typelst.append(f"{l}samples_repetition{r}")

    for j in range(len(lst)):
        oneTestIteration(lst[j], typelst[j])
        print(f"---- {typelst[j]} finished ----")
