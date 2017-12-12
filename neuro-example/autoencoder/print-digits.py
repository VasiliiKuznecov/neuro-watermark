# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
import matplotlib.pyplot as plt

import math

from keras.models import load_model

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

def get_closest_half(value, bit):
    (fractional_part, integer_part) = math.modf(value)

    if integer_part % 2 == bit:
        return integer_part + .5

    candidats = [integer_part - 1 + .5, integer_part + 1 + .5]
    diffs = np.absolute(candidats - value)

    return candidats[np.argmin(diffs)]


def get_injection(code):
    injected_bits = []

    for i in xrange(len(code)):
       (fractional_part, integer_part) = math.modf(code[i])

       injected_bits.append(int(integer_part % 2))

    return injected_bits


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test  = x_test .astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test  = np.reshape(x_test,  (len(x_test),  28, 28, 1))

encoder = load_model('neuro-example/autoencoder/encoder.h5')
decoder = load_model('neuro-example/autoencoder/decoder.h5')

n = 1

bits = np.random.random_integers(0, 1, 49)

print bits

imgs = x_test[:n]
encoded_imgs = encoder.predict(imgs, batch_size=n)

# print encoded_imgs[0]

for i in xrange(n):
    for j in xrange(len(encoded_imgs[i])):
        encoded_imgs[i][j] = get_closest_half(encoded_imgs[i][j], bits[j])

    print get_injection(encoded_imgs[i])

# print encoded_imgs[0]

decoded_imgs = decoder.predict(encoded_imgs, batch_size=n)

double_encoded_imgs = encoder.predict(decoded_imgs, batch_size=n)

# print double_encoded_imgs[0]

print get_injection(double_encoded_imgs[0])

count = 0
count2 = 0
inj1 = get_injection(encoded_imgs[0])
inj2 = get_injection(double_encoded_imgs[0])
for i in xrange(len(bits)):
    if inj1[i] != bits[i]:
        count += 1
    if inj1[i] != inj2[i]:
        count2 += 1

print count
print count2

# plot_digits(imgs, decoded_imgs)
