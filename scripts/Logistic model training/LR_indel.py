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


# Define useful functions
def mse(x, y):
    return ((x-y)**2).mean()

def corr(x, y):
    return np.corrcoef(x, y)[0, 1] ** 2

def onehotencoder(seq):
    nt= ['A','T','C','G']
    head = []
    l = len(seq)
    for k in range(l):
        for i in range(4):
            head.append(nt[i]+str(k))

    for k in range(l-1):
        for i in range(4):
            for j in range(4):
                head.append(nt[i]+nt[j]+str(k))
    head_idx = {}
    for idx,key in enumerate(head):
        head_idx[key] = idx
    encode = np.zeros(len(head_idx))
    for j in range(l):
        encode[head_idx[seq[j]+str(j)]] =1.
    for k in range(l-1):
        encode[head_idx[seq[k:k+2]+str(k)]] =1.
    return encode



# read the .txt from GitHub
# remove the guide sequence
# take apart the class labels
# TODO: is the test on teams seperate from train or is it included in the trainingdata
# insert in the xtrain etc.

Lindel_training = pd.read_csv("/home/cornelis/PycharmProjects/Lindel/data/Lindel_training.txt", sep='\t', header=None)
# column descriptions
seq_t   = Lindel_training.iloc[:, 0] # guide sequences
x_t     = Lindel_training.iloc[:, 1:3034] # 3033 binary features [2649 MH binary features + 384 one hot encoded features]
y_t     = Lindel_training.iloc[:, 3034:] # 557 observed outcome frequencies

x_t = x_t.astype('float32')
y_t = y_t.astype('float32')

train_size = round(len(x_t) * 0.9)
valid_size = round(len(x_t) * 0.1)

x = np.array(x_t)
y = np.array(y_t)

x_train,x_valid = np.array(x[:train_size,:]),np.array(x[train_size:,:])
y_train,y_valid = np.array(y[:train_size,:]),np.array(y[train_size:,:])

# Train model
# lambdas = 10 ** np.arange(-10, -1, 0.1)
# print(lambdas)
errors_l1, errors_l2 = [], []
lambdas = 10 ** np.arange(-10, -1, 1.0)
errors_l1, errors_l2 = [], []
for l in lambdas:
    np.random.seed(0)
    model = Sequential()
    model.add(Dense(557,  activation='softmax', input_shape=(3033,), kernel_regularizer=l2(l)))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mse'])
    model.fit(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid),
              callbacks=[EarlyStopping(patience=1)], verbose=1)
    y_hat = model.predict(x_valid)
    errors_l2.append(mse(y_hat, y_valid))

for l in lambdas:
    np.random.seed(0)
    model = Sequential()
    model.add(Dense(557,  activation='softmax', input_shape=(3033,), kernel_regularizer=l1(l)))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])
    model.fit(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid),
              callbacks=[EarlyStopping(patience=1)], verbose=1)
    y_hat = model.predict(x_valid)
    errors_l1.append(mse(y_hat, y_valid))


# np.save(workdir+'mse_l1_indel.npy',errors_l1)
# np.save(workdir+'mse_l2_indel.npy',errors_l2)

# final model
l = lambdas[np.argmin(errors_l1)]
np.random.seed(0)
model = Sequential()
model.add(Dense(557, activation='softmax', input_shape=(3033,), kernel_regularizer=l1(l)))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])
history = model.fit(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid),
          callbacks=[EarlyStopping(patience=1)], verbose=0)

model.save('./L1_indel.h5')
#
#
l = lambdas[np.argmin(errors_l2)]
np.random.seed(0)
model = Sequential()
model.add(Dense(557, activation='softmax', input_shape=(3033,), kernel_regularizer=l2(l)))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])
history = model.fit(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid),
          callbacks=[EarlyStopping(patience=1)], verbose=0)

model.save('./L2_indel.h5')
