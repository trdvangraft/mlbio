from mainTraining import main as maintraining
from visualizations.visMSE import generateOutcomes

def oneTestIteration(traingset, trainingtype):
    maintraining(traingset)
    generateOutcomes(trainingtype)

if __name__ == "__main__":
    levels = [0.1,0.2, 0.3, 0.4, 0.5]
    concentrations = [1, 0.1, 0.01]
    lst = []
    typelst = []
    for l in levels:
        for c in concentrations:
            lst.append(f"../data/Lindel_training_withnoise_{int(l*100)}_{int(c*100)}.txt")
            typelst.append(f"level{int(l*100)}_concentration{int(c*100)}")

    for j in range(len(lst)):
        oneTestIteration(lst[j], typelst[j])
        print(f"---- {typelst[j]} finished ----")


