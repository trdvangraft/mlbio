from keras.callbacks import EarlyStopping
from keras.layers import Dense, Input, Flatten
from keras.models import Sequential, load_model
from keras.regularizers import l2, l1


def model_creator(
    num_units,
    input_shape,
    kernel_regularizer=l2,
    kernel_weight=10**-4,
    activation="softmax",
    loss="categorical_crossentropy"
):
    model = Sequential()
    model.add(Dense(num_units,  activation=activation, input_shape=input_shape,
              kernel_regularizer=kernel_regularizer(kernel_weight)))
    model.compile(optimizer='adam',
                  loss=loss, metrics=['mse'])

    return model


def mse(x, y):
    return ((x-y)**2).mean()
