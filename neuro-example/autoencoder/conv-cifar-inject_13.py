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


input_img = Input(shape=(32, 32, 3))

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = Flatten()(x)
deinjected = Dense(2, activation='softmax')(x)

deinjector = Model(input_img, deinjected, name='deinjector')

injector = autoencoder

injected_img = injector(input_img)
output_injected = deinjector(injected_img)
output_not_injected = deinjector(input_img)

injection = Model(inputs=[input_img], outputs=[injected_img, output_injected], name='injection')
not_injection = Model(inputs=[input_img], outputs=[output_not_injected], name='not_injection')

injection.compile(optimizer='adam', loss=[sum_squared_error, 'categorical_crossentropy'], loss_weights=[1, 0.5])
not_injection.compile(optimizer='adam', loss='categorical_crossentropy', loss_weights=[0.5])

injection.summary()
not_injection.summary()


plot_model(injector, to_file='imgs/models/autoencoder/cifar_injector13.png', show_shapes=True)
plot_model(deinjector, to_file='imgs/models/autoencoder/cifar_not_injector13.png', show_shapes=True)
plot_model(injection, to_file='imgs/models/autoencoder/cifar_injection13.png', show_shapes=True)
plot_model(not_injection, to_file='imgs/models/autoencoder/cifar_not_injection13.png', show_shapes=True)

epoch_number = 60
batches_number = 1000
batch_size = 50

f1=open('neuro-example/autoencoder/cifar-13/stats.txt', 'w+')

for e in range(epoch_number):
    start_time = time.time()

    for i in range(batches_number):
        x_batch = x_train[i * batch_size : i * batch_size + batch_size]
        y_outputs_1 = np.repeat([[0, 1]], batch_size, axis=0)
        y_outputs_0 = np.repeat([[1, 0]], batch_size, axis=0)
        y_batch_injector = [x_batch, y_outputs_1]
        y_batch_not_injector = y_outputs_0

        if len(x_batch) != 0:
            loss_injector = injection.train_on_batch(x_batch, y_batch_injector)
            loss_not_injector = not_injection.train_on_batch(x_batch, y_batch_not_injector)

    end_time = time.time()
    print('epoch ' + str(e) + ' done')
    print(end_time - start_time, 's')
    print('loss inj: ')
    print(loss_injector)
    print('loss not_inj: ')
    print(loss_not_injector)

    f1.write('epoch ' + str(e) + ' done\n')
    f1.write('loss inj: \n')
    f1.write(str(loss_injector))
    f1.write('\n')
    f1.write('loss not_inj: \n')
    f1.write(str(loss_not_injector))
    f1.write('\n')
    f1.write('\n')

    injector.save('neuro-example/autoencoder/cifar-13/cifar_injector13-' + str(e) + '.h5')
    deinjector.save('neuro-example/autoencoder/cifar-13/cifar_deinjector13-' + str(e) + '.h5')
    injection.save('neuro-example/autoencoder/cifar-13/cifar_injection13-' + str(e) + '.h5')
    not_injection.save('neuro-example/autoencoder/cifar-13/cifar_not_injection13-' + str(e) + '.h5')

f1.close()
