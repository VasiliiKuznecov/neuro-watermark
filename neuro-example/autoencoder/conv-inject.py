# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

import seaborn as sns
import matplotlib.pyplot as plt

from keras.utils import plot_model


import math

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

encoder = load_model('neuro-example/autoencoder/c_encoder.h5')
encoder_copy = load_model('neuro-example/autoencoder/c_encoder.h5')
decoder = load_model('neuro-example/autoencoder/c_decoder.h5')

encoder_copy.name = 'encoder_copy'

for l in encoder.layers:
    l.trainable = False

for l in encoder_copy.layers:
    l.trainable = False

for l in decoder.layers:
    l.trainable = False

input_img = Input(shape=(28, 28, 1))

image_encoded = encoder(input_img)

flatten_code = Flatten()(image_encoded)
injected_code = Dense(7 * 7, activation='relu', name='injector')(flatten_code)

reshaped_injected_code = Reshape((7, 7, 1))(injected_code)
image_decoded_injected = decoder(reshaped_injected_code)
image_encoded_injected = encoder_copy(image_decoded_injected)
deinjector =  Dense(2, activation='softmax', name='deinjector')
output_bit_injected =deinjector(image_encoded_injected)

injector = Model(inputs=[input_img], outputs=[image_decoded_injected, output_bit_injected], name="injector")
injector.compile(optimizer='adam', loss='binary_crossentropy')
injector.summary()

output_bit_no_injection =deinjector(image_encoded)

not_injector = Model(inputs=[input_img], outputs=[output_bit_no_injection], name="not_injector")
not_injector.compile(optimizer='adam', loss='binary_crossentropy')
not_injector.summary()

plot_model(injector, to_file='imgs/models/autoencoder/injector.png', show_shapes=True)
plot_model(not_injector, to_file='imgs/models/autoencoder/not_injector.png', show_shapes=True)
