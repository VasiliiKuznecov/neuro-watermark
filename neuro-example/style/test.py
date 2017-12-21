import os
import time
import argparse

import numpy as np
from scipy.misc import imsave
from scipy.optimize import fmin_l_bfgs_b

from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg16


# Setup to receive command line arguments
parser = argparse.ArgumentParser(description='Keras implemenation of neural --style-- transfer (Gatys et al., 2015)')
parser.add_argument('content_image_path', metavar='content_image', type=str, help='Path to the content image to transform.')
parser.add_argument('--output_image_path', metavar='output_path', type=str, default='transformed', help='Path to store the generated image.')
parser.add_argument('--content_weight', type=float, default=0.025, help='Set content loss weight.')
parser.add_argument('--per_pixel_weight', help='Set --style-- loss weight.', type=float, default=1.0)
parser.add_argument('--total_variation_weight', help='Set total variation loss weight.', type=float, default=1.0)
parser.add_argument('--width', help='Set width of the generated image.', type=int, default=512)
parser.add_argument('--height', help='Set height of the generated image.', type=int, default=512)
parser.add_argument('--iterations', help='Set the number of iterations to run the optimizer for.', type=int, default=20)

# Parse command line arguments
args = parser.parse_args()
content_image_path = args.content_image_path
output_image_path = args.output_image_path
content_weight = args.content_weight
per_pixel_weight = args.per_pixel_weight
total_variation_weight = args.total_variation_weight
img_nrows = args.height
img_ncols = args.width
assert img_ncols == img_nrows, 'Due to the use of the Gram matrix, width and height must match.'
iterations = args.iterations

# Utility function to open, resize and format pictures into appropriate tensors
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68

    return img

# Utility function to convert a tensor into a valid image
def deprocess_image(x):
    if K.image_dim_ordering() == 'th':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))
    x = x[:, :, ::-1]
    # x[:, :, 0] += 103.939
    # x[:, :, 1] += 116.779
    # x[:, :, 2] += 123.68
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# Get tensor representations of our images
base_image = K.variable(preprocess_image(content_image_path))


# TODO: Start the combination image as the original image?
# This will contain our generated image
if K.image_dim_ordering() == 'th':
    combination_image = K.placeholder((1, 3, img_nrows, img_ncols))
else:
    combination_image = K.placeholder((1, img_nrows, img_ncols, 3))

# Combine the 3 images into a single Keras tensor
input_tensor = K.concatenate([base_image,
                              combination_image], axis=0)

# Build the VGG16 network with our 3 images as input
# the model will be loaded with pre-trained ImageNet weights
model = vgg16.VGG16(input_tensor=input_tensor,
                    weights='imagenet', include_top=False)
print('Model loaded.')

# Get the symbolic outputs of each "key" layer (we gave them unique names).
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

def per_pixel_loss(base, combination):
    return K.sum(K.square(combination - base))

# An auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image
def content_loss(base, combination):
    return K.sum(K.square(combination - base))

# The 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent
def total_variation_loss(x):
    assert K.ndim(x) == 4
    if K.image_dim_ordering() == 'th':
        a = K.square(x[:, :, :img_nrows-1, :img_ncols-1] - x[:, :, 1:, :img_ncols-1])
        b = K.square(x[:, :, :img_nrows-1, :img_ncols-1] - x[:, :, :img_nrows-1, 1:])
    else:
        a = K.square(x[:, :img_nrows-1, :img_ncols-1, :] - x[:, 1:, :img_ncols-1, :])
        b = K.square(x[:, :img_nrows-1, :img_ncols-1, :] - x[:, :img_nrows-1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

# Combine these loss functions into a single scalar
loss = K.variable(0.)
layer_features = outputs_dict['block2_conv2']
base_image_features = layer_features[0, :, :, :]
combination_features = layer_features[1, :, :, :]
loss += content_weight * content_loss(base_image_features,
                                      combination_features)

loss += per_pixel_weight * per_pixel_loss(base_image, combination_image)
loss += total_variation_weight * total_variation_loss(combination_image)

# Get the gradients of the generated image wrt the loss
grads = K.gradients(loss, combination_image)

outputs = [loss]
if type(grads) in {list, tuple}:
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([combination_image], outputs)

def eval_loss_and_grads(x):
    if K.image_dim_ordering() == 'th':
        x = x.reshape((1, 3, img_nrows, img_ncols))
    else:
        x = x.reshape((1, img_nrows, img_ncols, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

# This Evaluator class makes it possible
# to compute loss and gradients in one pass
# while retrieving them via two separate functions,
# "loss" and "grads". This is done because scipy.optimize
# requires separate functions for loss and gradients,
# but computing them separately would be inefficient.
class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

# Run scipy-based optimization (L-BFGS) over the pixels of the generated image
# so as to minimize the neural --style-- loss
if K.image_dim_ordering() == 'th':
    x = np.random.uniform(0, 255, (1, 3, img_nrows, img_ncols)) - 128.
else:
    x = np.random.uniform(0, 255, (1, img_nrows, img_ncols, 3)) - 128.

for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    # save current generated image
    img = deprocess_image(x.copy())
    fname = os.path.join(output_image_path,
                         'test_c_%s_cw_%g_ppw_%g_tvw_%g_i_%d.png' %
                         (os.path.splitext(os.path.basename(content_image_path))[0],
                          content_weight,
                          per_pixel_weight,
                          total_variation_weight,
                          i))
    imsave(fname, img)
    end_time = time.time()
    print('Image saved as', fname)
    print('Iteration %d completed in %ds' % (i, end_time - start_time))
