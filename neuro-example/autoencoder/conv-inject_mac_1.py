# -*- coding: utf- -*-
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

import time

import seaborn as sns
import matplotlib.pyplot as plt

from keras.utils import plot_model

import math
import random

import keras

from keras.models import load_model

from keras.layers import Input, Dense, Flatten, Reshape
from keras.models import Model

from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose


from keras.datasets import cifar10
import numpy as np

from keras import backend as K

def sum_squared_error(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true), axis=-1)

def plot_digits(*args):
    args = [x.squeeze() for x in args]
    n = min([x.shape[0] for x in args])

    plt.figure(figsize=(2*n, 2*len(args)))
    for j in range(n):
        for i in range(len(args)):
            ax = plt.subplot(len(args), n, i*n + j + 1)
            plt.imshow(args[i][j])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.show()

def create_deep_conv_ae():
    input_img = Input(shape=(32, 32, 3))

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    encoded = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

    input_encoded = Input(shape=(32, 32, 32))
    x = Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(input_encoded)
    decoded = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)

    # Модели
    encoder = Model(input_img, encoded, name="encoder")
    decoder = Model(input_encoded, decoded, name="decoder")
    autoencoder = Model(input_img, decoder(encoder(input_img)), name="autoencoder")
    return encoder, decoder, autoencoder

encoder, decoder, autoencoder = create_deep_conv_ae()

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32') / 255.
x_test  = x_test .astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 32, 32, 3))
x_test  = np.reshape(x_test,  (len(x_test),  32, 32, 3))

autoencoder.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss=sum_squared_error, loss_weights=[1])


epoch_number = 60
batches_number = 1000
batch_size = 50

autoencoder.fit(x_train, x_train,
                epochs=12,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

autoencoder.save('neuro-example/autoencoder/cifar-mac-1/autoencoder.h5')
