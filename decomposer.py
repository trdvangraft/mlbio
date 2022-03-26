import numpy as np
from src.modelTraining.deletion import DeletionModel
from src.modelTraining.indel import InDelModel
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from src.modelTraining.insertion import InsertionModel


def plotExplainedPCA():
    indel = InDelModel()
    x_train, x_test, y_train, y_test = indel.prepare_data()

    low = 1
    high = 300

    x = range(low,high)
    y = []

    for i in range(low,high):
        pca = PCA(n_components=i)
        pca.fit(x_train)

        # print('Explained {} divided over variation per principal component: {}'.format(sum(pca.explained_variance_ratio_), pca.explained_variance_ratio_))
        # print("Explained variation {} by {} components".format(round(sum(pca.explained_variance_ratio_),3), i))
        y.append(sum(pca.explained_variance_ratio_))

    plt.plot(x, y)
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Number of components')
    plt.title('Explained variance ratio vs number of components')
    plt.show()

def genPCAComponents(numComponents = 220):
    indel = InDelModel()
    x_train, x_test, y_train, y_test = indel.prepare_data()

    pca = PCA(n_components=numComponents)
    pca.fit(x_train)

    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)

    return x_train, x_test, y_train, y_test

def save_errors(errors, name):
    reg_names = ["l1", "l2"]
    for reg_name, error in zip(reg_names, errors):
        np.save(f"./results/{name}_{reg_name}.npy", error)



if __name__ == "__main__":
    # plotExplainedPCA()

    x_train, x_test, y_train, y_test = genPCAComponents(220)
    print(x_train.shape)

    del_model = DeletionModel()
    ins_model = InsertionModel()
    indel_model = InDelModel()

    print("---- TRAINING: Indel model ----")
    indel_errors = indel_model.train_model()
    save_errors(indel_errors, "indel")
    print("---- TRAINING: Insertion model ----")
    ins_errors = ins_model.train_model()
    save_errors(ins_errors, "insertion")
    print("---- TRAINING: Deletion model ----")
    del_errors = del_model.train_model()
    save_errors(del_errors, "deletion")