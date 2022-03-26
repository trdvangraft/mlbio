import numpy as np

from src.modelTraining.base_model import BaseModel
from src.modelTraining.deletion import DeletionModel
from src.modelTraining.indel import InDelModel
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from src.modelTraining.insertion import InsertionModel


def plotExplainedPCA(arr, low = 1, high=200):
    x = range(low, high)
    y = []

    for i in range(low, high):
        pca = PCA(n_components=i)
        pca.fit(arr)

        # print('Explained {} divided over variation per principal component: {}'.format(sum(pca.explained_variance_ratio_), pca.explained_variance_ratio_))
        # print("Explained variation {} by {} components".format(round(sum(pca.explained_variance_ratio_),3), i))
        y.append(sum(pca.explained_variance_ratio_))

    plt.plot(x, y)
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Number of components')
    plt.title('Explained variance ratio vs number of components')
    plt.show()


def oneHotEncoder(seq):
    nt = ['A', 'T', 'C', 'G']

    encoded = []
    # One Hot Encode single nucleotides
    for i in range(len(seq)):
        s = seq[i]
        for j in nt:
            if s == j:
                encoded.append(1)
            else:
                encoded.append(0)
                # print("Target {} does not match template {}".format(s,j))

    # print("Len after mono-nuceotide is {}".format(len(encoded)))

    # One Hot Encode di-nucleotide
    for i in range(len(seq) - 1):
        s = seq[i:i+2]
        for j in nt:
            for k in nt:
                template = j + k
                if (s == template):
                    encoded.append(1)
                else:
                    encoded.append(0)
                    # print("Target {} does not match template {}".format(s,template))

    # print("Len after di-nuceotide is {}".format(len(encoded)))

    # One Hot Encode tri-nucleotide
    for i in range(len(seq) - 2):
        s = seq[i:i + 3]
        for j in nt:
            for k in nt:
                for l in nt:
                    template = j + k + l
                    if (s == template):
                        encoded.append(1)
                    else:
                        encoded.append(0)
                        # print("Target {} does not match template {}".format(s, template))

    # print("Len after tri-nuceotide is {}".format(len(encoded)))

    return encoded


def encodeAll(seqs):
    encodes = []
    for s in seqs:
        encodes.append(oneHotEncoder(s))

    return np.array(encodes)


def genPCAComponents(numComponents=220):
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

def lastNofElement(seqs, lim = 6):
    out = []
    for i in seqs:
        out.append(i[-lim:])

    return out

if __name__ == "__main__":
    # plotExplainedPCA()


    # print(a)

    # oneHotEncoder("AACTCG")

    model = BaseModel()
    s,x,y = model.get_sxy_split()
    print(s)
    s = lastNofElement(s)
    print(s)
    encodedX = encodeAll(s)
    print(encodedX)
    print(encodedX.shape)

    # x = x[:,-104:]
    # print(x.shape)

    train_size = round(x.shape[0] * 0.9)
    print(train_size)
    x_train, encodedX_test = encodedX[:train_size, :], encodedX[train_size:, :]
    # y_train, y_test = y[:train_size], y[train_size:]

    plotExplainedPCA(x_train,1,360)

    #
    # x_train, x_test, y_train, y_test = genPCAComponents(220)
    # print(x_train.shape)
    #
    # del_model = DeletionModel()
    # ins_model = InsertionModel()
    # indel_model = InDelModel()
    #
    # print("---- TRAINING: Indel model ----")
    # indel_errors = indel_model.train_model(x_train, x_test, y_train, y_test)
    # save_errors(indel_errors, "indel")
    # print(indel_errors)
    # print("---- TRAINING: Insertion model ----")
    # ins_errors = ins_model.train_model(x_train, x_test, y_train, y_test)
    # save_errors(ins_errors, "insertion")
    # print("---- TRAINING: Deletion model ----")
    # del_errors = del_model.train_model()
    # save_errors(del_errors, "deletion")
