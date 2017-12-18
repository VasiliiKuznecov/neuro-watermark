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


input_img = Input(shape=(28, 28, 1))

x = Conv2D(128, (7, 7), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (2, 2), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(1, (7, 7), activation='relu', padding='same')(x)
x = Flatten()(x)
deinjected = Dense(2, activation='softmax')(x)

deinjector = Model(input_img, deinjected, name='deinjector')

injector = load_model('neuro-example/autoencoder/c_autoencoder.h5')

injected_img = injector(input_img)
output_injected = deinjector(injected_img)
output_not_injected = deinjector(injected_img)

injection = Model(inputs=[input_img], outputs=[injected_img, output_injected], name='injection')
not_injection = Model(inputs=[input_img], outputs=[output_not_injected], name='not_injection')

plot_model(injector, to_file='imgs/models/autoencoder/injector3.png', show_shapes=True)
plot_model(deinjector, to_file='imgs/models/autoencoder/not_injector3.png', show_shapes=True)
plot_model(injection, to_file='imgs/models/autoencoder/injection3.png', show_shapes=True)
plot_model(not_injection, to_file='imgs/models/autoencoder/not_injection3.png', show_shapes=True)

epoch_number = 60
batches_number = 600
batch_size = 100

for e in range(epoch_number):
    for i in range(batches_number):
        x_batch = x_train[i * batch_size : i * batch_size + batch_size]
        y_outputs_1 = np.repeat([[0, 1]], batch_size, axis=0)
        y_outputs_0 = np.repeat([[1, 0]], batch_size, axis=0)
        y_batch_injector = [x_batch, y_outputs_1]
        y_batch_not_injector = y_outputs_0

        injection.train_on_batch(x_batch, y_batch_injector)
        not_injection.train_on_batch(x_batch, y_batch_not_injector)

    print('epoch ' + str(e) + ' done')

injector.save('neuro-example/autoencoder/injector3.h5')
deinjector.save('neuro-example/autoencoder/deinjector3.h5')
injection.save('neuro-example/autoencoder/injection3.h5')
deinjection.save('neuro-example/autoencoder/deinjection3.h5')
