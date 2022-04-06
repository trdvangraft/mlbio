# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
import pickle as pkl
import lightgbm as lgb


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = np.power(predictions - test_labels, 2)
    stderrors = np.std(errors)

    print('Model Performance')
    print('MSE Error: {:0.6f} degrees.'.format(errors.mean()))
    print('STD Error: {:0.6f} degrees.'.format(stderrors))


    return errors


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
    frame_shift = pkl.load(open('model_prereq.pkl', 'rb'))[3]

    labeltrain = []
    labeltest = []

    for el in obsp.to_numpy():
        labeltrain.append(np.dot(el, frame_shift))

    for el in obsptest:
        labeltest.append(np.dot(el, frame_shift))


    train = True
    filename = 'lgbmmodelFSreglessest.sav'
    if train:
        xs = onehot.to_numpy()
        ys = obsp.to_numpy()
        #data_dmatrix = xgb.DMatrix(data=feat, label=ys)
        params = {"bagging_frequency": 1}

        reg = lgb.LGBMRegressor(boosting_type="rf",
                                num_leaves=600,
                                n_estimators=100,
                                min_child_samples=1,
                                 subsample=.3,  # Standard RF bagging fraction
                                 reg_alpha=0,  # Hard L1 regularization
                                 reg_lambda=0,
                                 n_jobs=3,subsample_freq=2)

        reg.fit(feat, labeltrain)
        # pkl.dump(rf, open(filename, 'wb'))
    else:
        rf = pkl.load(open(filename, 'rb'))

    pkl.dump(reg, open('lgbmmodelFSbase.sav', 'wb'))


    base = pkl.load(open(filename, 'rb'))
    print(base.get_params())
    base_acc = evaluate(base, feattestn, np.array(labeltest))





# See PyCharm help at https://www.jetbrains.com/help/pycharm/
