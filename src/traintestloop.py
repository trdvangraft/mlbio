from mainTraining import main as maintraining
from visualizations.visMSE import generateOutcomes

def oneTestIteration(traingset, trainingtype):
    maintraining(traingset)
    generateOutcomes(trainingtype)

if __name__ == "__main__":
    levels = [20,40,60,80]
    repetitions = [0,1,2,3,4]
    lst = []
    typelst = []
    for l in levels:
        for r in repetitions:
            lst.append(f"../data/Lindel_training_bootstrapping_{l}_{r}.txt")
            typelst.append(f"{l}percent_repetition{r}")

    for j in range(len(lst)):
        oneTestIteration(lst[j], typelst[j])
        print(f"---- {typelst[j]} finished ----")


