# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
import matplotlib.pyplot as plt

import math
import keras

from keras.models import load_model

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
            # plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.show()

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32') / 255.
x_test  = x_test .astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 32, 32, 3))
x_test  = np.reshape(x_test,  (len(x_test),  32, 32, 3))

autoencoder = load_model('neuro-example/autoencoder/cifar_autoencoder4.h5')

keras.utils.print_summary(autoencoder)

print(autoencoder.get_config())

n = 10

imgs = x_test[:n]

decoded_imgs = autoencoder.predict(imgs, batch_size=n)

plot_digits(imgs, decoded_imgs)
