#!/usr/bin/env python

#System tools
import pickle as pkl
import os,sys,csv,re

from tqdm import tqdm_notebook as tqdm
# import pylab as pl
import numpy as np

from keras.callbacks import EarlyStopping
from keras.layers import Dense, Input, Flatten
from keras.models import Sequential, load_model
from keras.regularizers import l2, l1
import pandas as pd

if __name__ == "__main__":

    Lindel_training = pd.read_csv("/home/cornelis/PycharmProjects/Lindel/data/Lindel_training.txt", sep='\t', header=None)
    # column descriptions
    seq_t   = Lindel_training.iloc[:, 0] # guide sequences
    x_t     = Lindel_training.iloc[:, 2649:3033] # the full one hot encoding
    y_t     = Lindel_training.iloc[:, 3034:] # 557 observed outcome frequencies

    y_t = np.array(y_t)

    y_ins   = np.sum(y_t[:,-21:], axis=1)
    y_del   = np.sum(y_t[:,:-21], axis=1)

    y_t = np.array([[0, 1] if y_ins > y_del else [1, 0] for y_ins, y_del in zip(y_ins, y_del)])

    x_t = x_t.astype('float32')
    y_t = y_t.astype('float32')

    train_size = round(len(x_t) * 0.9)
    valid_size = round(len(x_t) * 0.1)

    x = np.array(x_t)
    y = np.array(y_t)

    x_train,x_valid = np.array(x[:train_size,:]),np.array(x[train_size:,:])
    y_train,y_valid = np.array(y[:train_size]),np.array(y[train_size:])

    np.random.seed(0)
    model = Sequential()
    # model.add(Dense(2,  activation='softmax', input_shape=(384,), kernel_regularizer=l2(10**-4)))
    model.add(Dense(900,  activation='relu', input_shape=(384,), kernel_regularizer=l2(10**-4)))
    model.add(Dense(800,  activation='sigmoid'))
    model.add(Dense(700,  activation='relu'))
    model.add(Dense(600,  activation='sigmoid'))
    model.add(Dense(2,  activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mse'])
    model.fit(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid),
              callbacks=[EarlyStopping(patience=1)], verbose=1)
    y_hat = model.predict(x_valid)