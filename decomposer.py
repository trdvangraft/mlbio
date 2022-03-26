from src.modelTraining.indel import InDelModel
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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



if __name__ == "__main__":
    plotExplainedPCA()