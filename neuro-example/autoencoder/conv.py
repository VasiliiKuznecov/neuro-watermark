#! /usr/bin/env python
# # -*- coding: utf-8 -*-
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


from keras.datasets import mnist
import numpy as np
from keras.utils import plot_model

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test  = x_test .astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test  = np.reshape(x_test,  (len(x_test),  28, 28, 1))

from keras.layers import Input, Dense, Flatten, Reshape
from keras.models import Model

from keras.layers import Conv2D, MaxPooling2D, UpSampling2D

def create_deep_conv_ae():
    input_img = Input(shape=(28, 28, 1))

    x = Conv2D(128, (7, 7), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (2, 2), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    encoded = Conv2D(1, (7, 7), activation='relu', padding='same')(x)

    # На этом моменте представление  (7, 7, 1) т.е. 49-размерное

    input_encoded = Input(shape=(7, 7, 1))
    x = Conv2D(32, (7, 7), activation='relu', padding='same')(input_encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (2, 2), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (7, 7), activation='sigmoid', padding='same')(x)

    # Модели
    encoder = Model(input_img, encoded, name="encoder")
    decoder = Model(input_encoded, decoded, name="decoder")
    autoencoder = Model(input_img, decoder(encoder(input_img)), name="autoencoder")
    return encoder, decoder, autoencoder

c_encoder, c_decoder, c_autoencoder = create_deep_conv_ae()
c_autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

c_encoder.summary()
c_decoder.summary()
c_autoencoder.summary()


plot_model(c_autoencoder, to_file='imgs/models/autoencoder/с_autoencoder.png', show_shapes=True)
plot_model(c_encoder, to_file='imgs/models/autoencoder/с_encoder.png', show_shapes=True)
plot_model(c_decoder, to_file='imgs/models/autoencoder/с_decoder.png', show_shapes=True)


# c_autoencoder.fit(x_train, x_train,
#                 epochs=64,
#                 batch_size=256,
#                 shuffle=True,
#                 validation_data=(x_test, x_test))

# c_autoencoder.save('neuro-example/autoencoder/c_autoencoder.h5')
# c_encoder.save('neuro-example/autoencoder/c_encoder.h5')
# c_decoder.save('neuro-example/autoencoder/c_decoder.h5')
