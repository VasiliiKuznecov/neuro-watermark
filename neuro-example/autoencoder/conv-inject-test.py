# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')

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

from keras.datasets import mnist
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

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test  = x_test .astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test  = np.reshape(x_test,  (len(x_test),  28, 28, 1))

injector = load_model('neuro-example/autoencoder/injector2.h5')
not_injector = load_model('neuro-example/autoencoder/not_injector2.h5')

n = 10

imgs = x_test[:n]
injected_imgs = injector.predict(imgs, batch_size=n)

print(injected_imgs[1])

not_injected_imgs = not_injector.predict(imgs, batch_size=n)

print(not_injected_imgs)

injected_imgs_2 = not_injector.predict(injected_imgs[0], batch_size=n)

print(injected_imgs_2)

plot_digits(imgs, injected_imgs[0])
