#! /usr/bin/env python
# # -*- coding: utf-8 -*-
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


from keras.datasets import cifar10
import numpy as np
from keras.utils import plot_model

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32') / 255.
x_test  = x_test .astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 32, 32, 3))
x_test  = np.reshape(x_test,  (len(x_test),  32, 32, 3))

from keras.layers import Input, Dense, Dropout, Flatten, Reshape
from keras.models import Model

from keras.layers import Conv2D, MaxPooling2D, UpSampling2D

def create_deep_conv_ae():
    input_img = Input(shape=(32, 32, 3))

    x = Conv2D(32, (2, 2), activation='relu', padding='same')(input_img)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (2, 2), activation='relu', padding='same')(x)
    x = Conv2D(3, (8, 8), activation='sigmoid', padding='same')(x)

    # Модели
    autoencoder = Model(input_img, x, name="autoencoder")
    return autoencoder

autoencoder = create_deep_conv_ae()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.summary()

plot_model(autoencoder, to_file='imgs/models/autoencoder/сifar_autoencoder3.png', show_shapes=True)

autoencoder.fit(x_train, x_train,
                epochs=64,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

autoencoder.save('neuro-example/autoencoder/cifar_autoencoder3.h5')
