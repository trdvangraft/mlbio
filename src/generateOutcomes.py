import numpy as np

from predictor import gen_prediction, write_file
import pickle as pkl
import pandas as pd

from keras.models import load_model


prerequesites = pkl.load(open("../data/model_prereq.pkl", 'rb'))
Lindel_training = pd.read_csv("../data/Lindel_test.txt", sep='\t', header=None)
# column descriptions
gseq = Lindel_training.iloc[:, 0]  # guide sequences
feat = Lindel_training.iloc[:, 1:3034]  # 3033 binary features [2649 MH binary features + 384 one hot encoded features]
hencode = feat.iloc[:, -384:] # One hot encoded features
ins = hencode.iloc[:, -104:] # Insertion features
obsp = Lindel_training.iloc[:,3034:]  # 557 observed outcome frequencies

predictions = []
observations = []

for i in range(0, 440):
    mode = "l2"
    if mode == "l2":
        indel, deletion, insertion = load_model("../models/indel_l2.h5"), load_model("../models/deletion_l2.h5"), load_model("../models/insertion_l2.h5")
    else:
        indel, deletion, insertion = load_model("../models/indel_l1.h5"), load_model("../models/deletion_l1.h5"), load_model("../models/insertion_l1.h5")

    # Generate the predicted value and frameshift
    y_hat, fs = gen_prediction(hencode.iloc[i, :].to_numpy(), ins.iloc[i, :].to_numpy(), feat.iloc[i, :].to_numpy(),
                               prerequesites, indel, deletion, insertion)

    predictions.append(y_hat)

    # Get the observed value
    y_obs = obsp.iloc[i, :].to_numpy()

    observations.append(y_obs)

filePredicted = open("predictedLabels.txt", 'w')
filePredicted.write(str(predictions))
fileTrue = open("trueLabels.txt", 'w')
fileTrue.write(str(observations))
