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

    x = Conv2D(128, (8, 8), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (2, 2), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    encoded = Conv2D(3, (8, 8), activation='relu', padding='same')(x)

    input_encoded = Input(shape=(8, 8, 3))
    x = Conv2D(32, (8, 8), activation='relu', padding='same')(input_encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (2, 2), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (8, 8), activation='sigmoid', padding='same')(x)

    # Модели
    encoder = Model(input_img, encoded, name="encoder")
    decoder = Model(input_encoded, decoded, name="decoder")
    autoencoder = Model(input_img, decoder(encoder(input_img)), name="autoencoder")
    return encoder, decoder, autoencoder

encoder, decoder, autoencoder = create_deep_conv_ae()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

encoder.summary()
decoder.summary()
autoencoder.summary()

plot_model(autoencoder, to_file='imgs/models/autoencoder/сifar_autoencoder.png', show_shapes=True)
plot_model(encoder, to_file='imgs/models/autoencoder/сifar_encoder.png', show_shapes=True)
plot_model(decoder, to_file='imgs/models/autoencoder/сifar_decoder.png', show_shapes=True)

autoencoder.fit(x_train, x_train,
                epochs=64,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

autoencoder.save('neuro-example/autoencoder/cifar_autoencoder.h5')
encoder.save('neuro-example/autoencoder/cifar_encoder.h5')
decoder.save('neuro-example/autoencoder/cifar_decoder.h5')
