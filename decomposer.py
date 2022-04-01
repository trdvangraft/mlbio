import numpy
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
        print("Explained variation {} by {} components".format(round(sum(pca.explained_variance_ratio_),3), i))
        y.append(sum(pca.explained_variance_ratio_))

    plt.plot(x, y)
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Number of components')
    plt.title('Explained variance ratio vs number of components')
    plt.show()


def oneHotEncoder(seq, trinucleotide = False):
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

    if not trinucleotide:
        return encoded

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


def encodeAll(seqs, trinucleotide = False):
    encodes = []
    for s in seqs:
        encodes.append(oneHotEncoder(s,trinucleotide))

    return np.array(encodes)


def genPCAComponents(x_train, x_test, numComponents=220):

    pca = PCA(n_components=numComponents)
    pca.fit(x_train)

    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)

    return x_train, x_test


def save_errors(errors, name):
    reg_names = ["l1", "l2"]
    for reg_name, error in zip(reg_names, errors):
        np.save(f"./results/{name}_{reg_name}.npy", error)

def lastNofElement(seqs, lim = 6):
    out = []
    for i in seqs:
        out.append(i[-lim:])

    return out

def prepare_data(self):
    x_t, y_t = self.get_xy_split()

    x_t = x_t[:, -384:]

    y_ins = np.sum(y_t[:, -21:], axis=1)
    y_del = np.sum(y_t[:, :-21], axis=1)

    y_t = np.array([[0, 1] if y_ins > y_del else [1, 0]
                    for y_ins, y_del in zip(y_ins, y_del)]).astype('float32')

    train_size = round(len(x_t) * 0.9)

    x_train, x_test = x_t[:train_size, :], x_t[train_size:, :]
    y_train, y_test = y_t[:train_size], y_t[train_size:]

    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    model = BaseModel()
    s,x,y = model.get_sxy_split()
    # print(s)
    s = lastNofElement(s)
    # print(s)
    encodedX = encodeAll(s, trinucleotide=True)
    # print(encodedX)
    # print(encodedX.shape)

    # SET WHICH ENCODING IS SET(default is from lindel trainingset
    x = encodedX;

    train_size = round(x.shape[0] * 0.9)
    x_train, x_test = encodedX[:train_size, :], encodedX[train_size:, :]
    y_train, y_test = y[:train_size, :], y[train_size:, :]

    # Do PCA
    x_train,x_test = genPCAComponents(x_train,x_test,10)
    numpy.save("./data/insTriData104Test",x_test)
    print(x_test)
    print(x_test.shape)

    # y_ins = np.sum(y[:, -21:], axis=1)
    # y_del = np.sum(y[:, :-21], axis=1)
    #
    # y_indel = np.array([[0, 1] if y_ins > y_del else [1, 0]
    #                     for y_ins, y_del in zip(y_ins, y_del)]).astype('float32')
    # y_indel_train, y_indel_test = y_indel[:train_size, :], y_indel[train_size:, :]

    y_ins = y[:, -21:]
    y_train, y_test = y_ins[:train_size, :], y_ins[train_size:, :]
    # del_model = DeletionModel()
    ins_model = InsertionModel()
    # indel_model = InDelModel()

    # print("---- TRAINING: Indel model ----")
    # indel_errors = indel_model.train_model(x_train, x_test, y_indel_train, y_indel_test)
    # save_errors(indel_errors, "indel")
    # print(indel_errors)
    # print("---- TRAINING: Insertion model ----")
    # ins_errors = ins_model.train_model(x_train, x_test, y_train, y_test)
    # save_errors(ins_errors, "insertion")
    # print("---- TRAINING: Deletion model ----")
    # del_errors = del_model.train_model()
    # save_errors(del_errors, "deletion")
