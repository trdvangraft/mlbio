import pandas as pd
import os

def subsample(trainingset):
    data = pd.read_csv(trainingset, sep='\t', header=None)
    editData = data.copy()
    print(len(editData.index))

    df50 = editData.sample(frac=0.5)
    print(len(df50.index))
    filename50 = str(os.path.splitext(trainingset)[0]) + "_50%subsample" + ".txt"
    df50.to_csv(filename50, sep='\t', index=False, header=None)

    df75 = editData.sample(frac=0.75)
    print(len(df75.index))
    filename75 = str(os.path.splitext(trainingset)[0]) + "_75%subsample" + ".txt"
    df75.to_csv(filename75, sep='\t', index=False, header=None)

if __name__ == "__main__":
    subsample("Lindel_training.txt")
    # subsample("./Lindel_training_withnoise_10.txt")
    # subsample("./Lindel_training_withnoise_20.txt")
    # subsample("./Lindel_training_withnoise_30.txt")
    # subsample("./Lindel_training_withnoise_40.txt")
    # subsample("./Lindel_training_withnoise_50.txt")


