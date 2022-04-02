import os, sys
from src.predictor import gen_prediction, write_file
import pickle as pkl
import pandas as pd
from src.modelTraining.model_factory import mse
import numpy as np


from keras.models import load_model


prerequesites = pkl.load(open("C:\\Users\\corne\\PycharmProjects\\mlbio\\data\\model_prereq.pkl", 'rb'))
Lindel_training = pd.read_csv("C:\\Users\\corne\\PycharmProjects\\mlbio\\data\\Lindel_test.txt", sep='\t', header=None)
# column descriptions
gseq = Lindel_training.iloc[:, 0]  # guide sequences
feat = Lindel_training.iloc[:, 1:3034]  # 3033 binary features [2649 MH binary features + 384 one hot encoded features]
hencode = feat.iloc[:, -384:] # One hot encoded features
ins = np.load("C:\\Users\\corne\\PycharmProjects\\mlbio\\data\\insTest.npy") # Insertion features
obsp = Lindel_training.iloc[:,3034:]  # 557 observed outcome frequencies

msearray = []
fsarray = []
boxarray = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]

# Get frame shift array from prerequesites
frame_shift = prerequesites[3]

for i in range(0, 440):
    mode = "l2"
    if mode == "l2":
        indel, deletion, insertion = load_model("C:\\Users\\corne\\PycharmProjects\\mlbio\\models\\indel_l2.h5"), load_model("C:\\Users\\corne\\PycharmProjects\\mlbio\\models\\deletion_l2.h5"), load_model("C:\\Users\\corne\\PycharmProjects\\mlbio\\models\\insertion_l2.h5")
    else:
        indel, deletion, insertion = load_model("../models/indel_l1.h5"), load_model("../models/deletion_l1.h5"), load_model("../models/insertion_l1.h5")

    # Generate the predicted value and frameshift
    y_hat, fs = gen_prediction(hencode.iloc[i, :].to_numpy(), ins[i, :], feat.iloc[i, :].to_numpy(),
                               prerequesites, indel, deletion, insertion)
    # Get the observed value
    y_obs = obsp.iloc[i, :].to_numpy()

    # Calculate observed frame shift
    y_obs_fs = np.dot(y_obs, frame_shift)
    # Array with first observed frame shift and then predicted frameshift
    fsarray.append([y_obs_fs, fs])

    # Calculate the MSE and append it to the msearray
    res = mse(y_obs, y_hat)
    msearray.append(res)

    # Calculates values for boxplots
    # Add for every label the difference between the observed and the predicted probability
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


print(fsarray)
print(boxarray)


# Write every array to txt file
file = open("testMSE.txt", 'w')
file.write(str(msearray))
file.close()

file2 = open("testFS.txt", 'w')
file2.write(str(fsarray))
file2.close()

file3 = open("testBox.txt", 'w')
file3.write(str(boxarray))
file3.close()