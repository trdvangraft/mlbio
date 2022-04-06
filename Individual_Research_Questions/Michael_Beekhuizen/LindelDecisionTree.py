# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = np.power(predictions - test_labels, 2).mean(axis=1)
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

    av = obsp.to_numpy().mean(axis=0)
    avg = av / av.sum()
    print(avg.sum())
    train = True
    filename = 'randommodel.sav'
    if train:
        xs = onehot.to_numpy()
        ys = obsp.to_numpy()
        dt = DecisionTreeRegressor()

        # Number of features to consider at every split
        max_features = ['auto','sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(1, 50, num=20)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 4, 6, 8, 10, 12]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 3, 4]

        random_grid = {'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf}

        dt_random = RandomizedSearchCV(estimator=dt, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                       n_jobs=-1)
        dt_random.fit(feat, ys)
        print(dt_random.best_params_)
        #dt.fit(xs, ys)
        # pkl.dump(rf, open(filename, 'wb'))
    else:
        rf = pkl.load(open(filename, 'rb'))


    best = dt_random.best_estimator_
    # best = pkl.load(open('randommodelallf4.sav', 'rb'))
    random_acc = evaluate(best, feattestn, obsptest)

    pkl.dump(best, open('decisiontreerandom.sav', 'wb'))
    #print(rf.get_params())
