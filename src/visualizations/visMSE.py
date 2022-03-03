from src.predictor import gen_prediction, write_file
import pickle as pkl
import pandas as pd
from src.modelTraining.model_factory import mse
import numpy as np
import os, sys

print(os.getcwd())
prerequesites = pkl.load(open("../data/model_prereq.pkl", 'rb'))
Lindel_training = pd.read_csv("../data/Lindel_test.txt", sep='\t', header=None)
# column descriptions
gseq = Lindel_training.iloc[:, 0]  # guide sequences
feat = Lindel_training.iloc[:, 1:3034]  # 3033 binary features [2649 MH binary features + 384 one hot encoded features]
hencode = feat.iloc[:, -384:]
ins = hencode.iloc[:, -104:]
obsp = Lindel_training.iloc[:,3034:]  # 557 observed outcome frequencies

msearray = []
fsarray = []

frame_shift = prerequesites[3]

for i in range(0, 440):
    y_hat, fs = gen_prediction(hencode.iloc[i, :].to_numpy(), ins.iloc[i, :].to_numpy(), feat.iloc[i, :].to_numpy(),
                               prerequesites)
    y_obs = obsp.iloc[i, :].to_numpy()
    y_obs_fs = np.dot(y_obs, frame_shift)
    #res = mse(y_hat, y_obs)
    #msearray.append(res)
    # Array with first observed frame shift and then predicted frameshift
    fsarray.append([y_obs_fs, fs])
    print(i)


file = open("testFS.txt", 'w')
file.write(str(fsarray))
file.close()