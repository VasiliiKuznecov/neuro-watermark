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

from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose

def create_deep_conv_ae():
    input_img = Input(shape=(32, 32, 3))

    x = Conv2D(16, (5, 5), activation='relu', padding='same')(input_img)
    x = Conv2D(32, (7, 7), activation='relu', padding='same')(x)
    encoded = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    input_encoded = Input(shape=(32, 32, 64))
    x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(input_encoded)
    x = Conv2DTranspose(16, (7, 7), activation='relu', padding='same')(x)
    decoded = Conv2DTranspose(3, (5, 5), activation='sigmoid', padding='same')(x)

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

plot_model(autoencoder, to_file='imgs/models/autoencoder/сifar_autoencoder4.png', show_shapes=True)
plot_model(encoder, to_file='imgs/models/autoencoder/сifar_encoder4.png', show_shapes=True)
plot_model(decoder, to_file='imgs/models/autoencoder/сifar_decoder4.png', show_shapes=True)

autoencoder.fit(x_train, x_train,
                epochs=64,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

autoencoder.save('neuro-example/autoencoder/cifar_autoencoder4.h5')
encoder.save('neuro-example/autoencoder/cifar_encoder4.h5')
decoder.save('neuro-example/autoencoder/cifar_decoder4.h5')
