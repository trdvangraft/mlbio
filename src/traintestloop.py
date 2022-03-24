from mainTraining import main as maintraining
from visualizations.visMSE import generateOutcomes

def oneTestIteration(traingset, trainingtype):
    maintraining(traingset)
    generateOutcomes(trainingtype)

if __name__ == "__main__":
    datadir = "../data/Lindel_training_"
    type = "withnoise_10"
    trainingfile = datadir + type + ".txt"
    oneTestIteration(trainingfile, type)


