import numpy as np
import pandas as pd


class BaseModel:
    def __init__(self,trainingset):
        self.trainingset = trainingset

    def get_lambda(self):
        return 10.0 ** np.arange(-10, -1, 1)

    def get_xy_split(self):
        data = pd.read_csv(
            self.trainingset, sep='\t', header=None)
        x_t = data.iloc[:, 1:3034]  # the full one hot encoding
        # 557 observed outcome frequencies
        y_t = data.iloc[:, 3034:]
        return np.array(x_t), np.array(y_t)
