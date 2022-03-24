import os

import numpy as np

from src.modelTraining.base_model import BaseModel
from src.modelTraining.model_factory import model_creator, mse

from keras.callbacks import EarlyStopping
from keras.regularizers import l2, l1


class InDelModel(BaseModel):
    def __init__(self,trainingset):
        super().__init__(trainingset=trainingset)

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

    def train_model(self):
        x_train, x_test, y_train, y_test = self.prepare_data()

        np.random.seed(0)
        lambdas = self.get_lambda()

        models_l1, models_l2 = [], []
        errors_l1, errors_l2 = [], []

        for idx, kernel_weight in enumerate(lambdas):
            print(f"Percentage done: ({idx/len(lambdas):.2f})")

            model_l1 = model_creator(
                num_units=2,
                kernel_regularizer=l1,
                kernel_weight=kernel_weight,
                loss="binary_crossentropy",
                input_shape=(384, )
            )
            model_l1.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test),
                         callbacks=[EarlyStopping(patience=1)], verbose=0)
            models_l1.append(model_l1)
            y_hat = model_l1.predict(x_test)
            errors_l1.append(mse(y_hat, y_test))

            model_l2 = model_creator(
                num_units=2,
                kernel_regularizer=l2,
                kernel_weight=kernel_weight,
                loss="binary_crossentropy",
                input_shape=(384, )
            )
            model_l2.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test),
                         callbacks=[EarlyStopping(patience=1)], verbose=0)
            models_l2.append(model_l2)
            y_hat = model_l2.predict(x_test)
            errors_l2.append(mse(y_hat, y_test))

        models_l1[np.argmin(errors_l1)].save("./models/indel_l1.h5")
        models_l2[np.argmin(errors_l2)].save("./models/indel_l2.h5")

        return errors_l1, errors_l2
