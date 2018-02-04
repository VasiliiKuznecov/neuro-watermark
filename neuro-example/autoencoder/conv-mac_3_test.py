# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

import seaborn as sns
import matplotlib.pyplot as plt

from keras.utils import plot_model


import math
import random

import keras

from keras.models import load_model

from keras.layers import Input, Dense, Flatten, Reshape
from keras.models import Model

from keras.layers import Conv2D, MaxPooling2D, UpSampling2D

from keras import backend as K

def sum_squared_error(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true), axis=-1)

keras.losses.sum_squared_error = sum_squared_error

import keras.losses

from keras.datasets import cifar10
import numpy as np

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

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32') / 255.
x_test  = x_test .astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 32, 32, 3))
x_test  = np.reshape(x_test,  (len(x_test),  32, 32, 3))

injector = load_model('neuro-example/autoencoder/cifar-mac-3/cifar_injector3-114.h5')
deinjector = load_model('neuro-example/autoencoder/cifar-mac-3/cifar_deinjector3-114.h5')
injection = load_model('neuro-example/autoencoder/cifar-mac-3/cifar_injection3-114.h5')
not_injection = load_model('neuro-example/autoencoder/cifar-mac-3/cifar_not_injection3-114.h5')

n = 10

imgs = x_test[:n]
injected_imgs = injection.predict(imgs, batch_size=n)

print(injected_imgs[1])

not_injected_imgs = not_injection.predict(imgs, batch_size=n)

print(not_injected_imgs)

injected_imgs_2 = not_injection.predict(injected_imgs[0], batch_size=n)

print(injected_imgs_2)

plot_digits(imgs, injected_imgs[0])
