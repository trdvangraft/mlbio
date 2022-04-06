# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
import pickle as pkl
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = np.power(predictions - test_labels, 2).mean(axis=1)
    prerequesites = pkl.load(open("./model_prereq.pkl", 'rb'))
    fsarray = []
    frame_shift = prerequesites[3]
    for i in range(0, 440):
        ofs = np.dot(test_labels[i,:], frame_shift)
        nfs = np.dot(predictions[i, :], frame_shift)
        fsarray.append([ofs, nfs])
    stderrors = np.std(errors)

    print('Model Performance')
    print('MSE Error: {:0.6f} degrees.'.format(errors.mean()))
    print('STD Error: {:0.6f} degrees.'.format(stderrors))


    return errors, fsarray

def boxPlotdata(model, test_features, test_labels):
    predictions = model.predict(test_features)
    boxarray = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    for i in range(0, 440):
        y_hat = predictions[i]
        y_obs = test_labels[i]
        boxarray[0].append(y_obs[424] - y_hat[424])
        boxarray[1].append(y_obs[452] - y_hat[452])
        boxarray[2].append(y_obs[481] - y_hat[481])
        boxarray[3].append(y_obs[482] - y_hat[482])
        boxarray[4].append(y_obs[483] - y_hat[483])
        boxarray[5].append(y_obs[484] - y_hat[484])
        boxarray[6].append(y_obs[485] - y_hat[485])
        boxarray[7].append(y_obs[486] - y_hat[486])
        boxarray[8].append(y_obs[487] - y_hat[487])
        boxarray[9].append(y_obs[488] - y_hat[488])
        boxarray[10].append(y_obs[489] - y_hat[489])
        boxarray[11].append(y_obs[536] - y_hat[536])
        boxarray[12].append(y_obs[537] - y_hat[537])
        boxarray[13].append(y_obs[538] - y_hat[538])
        boxarray[14].append(y_obs[539] - y_hat[539])
        boxarray[15].append(y_obs[556] - y_hat[556])
    return boxarray

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    Lindel_training = pd.read_csv("Lindel_training.txt", sep='\t', header=None)
    Lindel_test = pd.read_csv("Lindel_test.txt", sep='\t', header=None)
    # column descriptions
    gseq = Lindel_training.iloc[:, 0]  # guide sequences
    feat = Lindel_training.iloc[:, 1:3034]  # 3033 binary features [2649 MH binary features + 384 one hot encoded features]
    onehot = feat.iloc[:, -384:] # One hot encoded features
    obsp = Lindel_training.iloc[:,3034:]  # 557 observed outcome frequencies

    feattest = Lindel_test.iloc[:, 1:3034]
    feattestn = feattest.to_numpy()
    feattestt = feattest.iloc[:, -384:].to_numpy()
    obsptest = Lindel_test.iloc[:, 3034:].to_numpy()  # 557 observed outcome frequencies

    mh_feat = Lindel_training.iloc[:, 1:2650]

    label, rev_index, features = pkl.load(open('feature_index_all.pkl', 'rb'))

    av = obsp.to_numpy().mean(axis=0)
    avg = av / av.sum()
    print(avg.sum())
    train = True
    filename = 'decisiontree.sav'

    base = pkl.load(open(filename, 'rb'))
    print(base.get_params())
    base_acc, fsarray = evaluate(base, feattestt, obsptest)
    boxplot = boxPlotdata(base, feattestt, obsptest)


    # Only for testing Random forest that is trained for Frameshift prediction

    # prerequesites = pkl.load(open("./model_prereq.pkl", 'rb'))
    # fsarray = []
    # frame_shift = prerequesites[3]
    # labeltest = []
    # for el in obsptest:
    #     labeltest.append(np.dot(el, frame_shift))
    #
    # fsbase = pkl.load(open('lgbmmodelFSreglessest.sav', 'rb'))
    # predictions = fsbase.predict(feattestn)


    # Only for getting the MSEs of the base performance model when averaging over every outcome

    # msearray = []
    # for i in range(0, 440):
    #     x = obsptest[i]
    #     res = ((x - avg) ** 2).mean()
    #     msearray.append(res)


    file = open("baseMSE.txt", 'w')
    file.write(str(base_acc))
    file.close()

    file2 = open("testFSlgbmreglessest.txt", 'w')
    file2.write(str(fsarray))
    file2.close()

    file3 = open("testBoxbaseRF.txt", 'w')
    file3.write(str(boxplot))
    file3.close()

    print("Test voltooid")



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
