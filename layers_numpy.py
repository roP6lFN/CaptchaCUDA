# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 18:07:15 2022

@author: Shaochang Liu
"""

import numpy as np
from functools import reduce
import math
from numba import cuda


def im2col(image, ksize, stride):
    # image [batchsize, width ,height, channel]
    image_col = []
    for i in range(0, image.shape[1] - ksize + 1, stride):
        for j in range(0, image.shape[2] - ksize + 1, stride):
            col = image[:, i:i + ksize, j:j + ksize, :].reshape([-1])
            image_col.append(col)
    image_col = np.array(image_col)
    return image_col


@cuda.jit
def im2col_gpu(x, col_image):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    i = cuda.blockIdx.x

    k = 0
    img_i = x[i]
    for l in range(tx, tx+3):
        for m in range(ty, ty+3):
            for n in range(img_i.shape[-1]):
                col_image[i, tx*(x.shape[2]-2)+ty, k] = img_i[l][m][n]
                k += 1


# padding='SAME'
class Conv2D(object):
    def __init__(self, shape, output_channels, stride=1, ksize=3):
        self.input_shape = shape  # [Batchsize, Width, Height, Channels]
        self.output_channels = output_channels
        self.input_channels = shape[-1]
        self.batchsize = shape[0]
        self.stride = stride
        self.ksize = ksize  # kernel size

        # initialization
        self.weights = np.random.standard_normal(
            (self.ksize, self.ksize, self.input_channels, self.output_channels)) / 100
        self.bias = np.random.standard_normal(self.output_channels) / 10

        # adam parameter
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.m_t_w = np.zeros(self.weights.shape)
        self.v_t_w = np.zeros(self.weights.shape)
        self.m_t_b = np.zeros(self.bias.shape)
        self.v_t_b = np.zeros(self.bias.shape)
        self.t = 0

        # gradient of BP
        self.eta = np.zeros(
            (shape[0], shape[1]//self.stride, shape[2]//self.stride, self.output_channels))
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)
        self.output_shape = self.eta.shape
    
    '''
    def forward(self, x):
        x = np.pad(x, (
            (0, 0), (self.ksize // 2, self.ksize // 2), (self.ksize // 2, self.ksize // 2), (0, 0)),
            'constant', constant_values=0)
        # im2col algorithm
        self.col_image = []
        for i in range(self.batchsize):
            img_i = x[i][np.newaxis, :]
            col_image_i = []
            for i in range(0, img_i.shape[1] - self.ksize + 1, self.stride):
                for j in range(0, img_i.shape[2] - self.ksize + 1, self.stride):
                    col = img_i[:, i:i + self.ksize,
                                j:j+self.ksize, :].reshape([-1])
                    col_image_i.append(col)
            self.col_image.append(col_image_i)
        self.col_image = np.array(self.col_image)
        # conv2d
        col_weights = self.weights.reshape([-1, self.output_channels])
        conv_out = np.zeros(self.eta.shape)
        for i in range(self.batchsize):
            col_conv_out = np.dot(self.col_image[i], col_weights) + self.bias
            conv_out[i] = np.reshape(col_conv_out, self.eta[0].shape)
        return conv_out
    '''
    
    def forward(self, x):
        col_weights = self.weights.reshape([-1, self.output_channels])
        x = np.pad(x, (
            (0, 0), (self.ksize // 2, self.ksize // 2), (self.ksize // 2, self.ksize // 2), (0, 0)),
            'constant', constant_values=0)
        self.col_image = []  # also use in BP
        conv_out = np.zeros(self.eta.shape)
        for i in range(self.batchsize):
            img_i = x[i][np.newaxis, :]
            col_image_i = im2col(img_i, self.ksize, self.stride) # Kaiming He's image to column algorithm
            conv_out[i] = np.reshape(
                np.dot(col_image_i, col_weights) + self.bias, self.eta[0].shape)
            self.col_image.append(col_image_i)
        self.col_image = np.array(self.col_image)
        return conv_out

    def forward_gpu(self, x):
        x = np.pad(x, (
            (0, 0), (self.ksize // 2, self.ksize // 2), (self.ksize // 2, self.ksize // 2), (0, 0)),
            'constant', constant_values=0)
        # im2col algorithm
        grid = (self.batchsize)
        block = (x.shape[1]-2, x.shape[2]-2)
        self.col_image = np.zeros((self.batchsize, (x.shape[1]-2)*(x.shape[2]-2),
                                   self.input_channels*(self.ksize**2)))
        im2col_gpu[grid, block](x, self.col_image)
        cuda.synchronize()
        # conv2d
        col_weights = self.weights.reshape([-1, self.output_channels])
        conv_out = np.zeros(self.eta.shape)
        for i in range(self.batchsize):
            col_conv_out = np.dot(self.col_image[i], col_weights) + self.bias
            conv_out[i] = np.reshape(col_conv_out, self.eta[0].shape)
        return conv_out

    def gradient(self, eta):
        # derivation
        self.eta = eta
        col_eta = np.reshape(eta, [self.batchsize, -1, self.output_channels])
        for i in range(self.batchsize):
            self.w_gradient += np.dot(self.col_image[i].T,
                                      col_eta[i]).reshape(self.weights.shape)
        self.b_gradient += np.sum(col_eta, axis=(0, 1))
        # padding
        pad_eta = np.pad(self.eta, (
            (0, 0), (self.ksize // 2, self.ksize // 2), (self.ksize // 2, self.ksize // 2), (0, 0)),
            'constant', constant_values=0)
        # rotate & transpose
        flip_weights = np.flipud(np.fliplr(self.weights))
        flip_weights = flip_weights.swapaxes(2, 3)
        col_flip_weights = flip_weights.reshape([-1, self.input_channels])
        col_pad_eta = np.array([im2col(
            pad_eta[i][np.newaxis, :], self.ksize, self.stride) for i in range(self.batchsize)])
        # as same as forward
        next_eta = np.dot(col_pad_eta, col_flip_weights)
        next_eta = np.reshape(next_eta, self.input_shape)
        return next_eta

    def adam(self, alpha=0.0001, weight_decay=0.0004, batch_size=32):
        # weight_decay = L2 regularization
        self.weights *= (1 - weight_decay)
        self.bias *= (1 - weight_decay)
        self.t += 1
        learning_rate_t = alpha * math.sqrt(
            1 - pow(self.beta2, self.t)) / (1 - pow(self.beta1, self.t))
        self.m_t_w = self.beta1 * self.m_t_w + (
            1 - self.beta1) * self.w_gradient / batch_size
        self.v_t_w = self.beta2 * self.v_t_w + (
            1 - self.beta2) * ((self.w_gradient / batch_size) ** 2)
        self.m_t_b = self.beta1 * self.m_t_b + (
            1 - self.beta1) * self.b_gradient / batch_size
        self.v_t_b = self.beta2 * self.v_t_b + (
            1 - self.beta2) * ((self.b_gradient / batch_size) ** 2)
        self.weights -= learning_rate_t * self.m_t_w / \
            (self.v_t_w + self.epsilon) ** 0.5
        self.bias -= learning_rate_t * self.m_t_b / \
            (self.v_t_b + self.epsilon) ** 0.5
        # zero gradient
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)


# padding='VALID'
class MaxPooling(object):
    def __init__(self, shape, ksize=2, stride=2):
        self.input_shape = shape
        self.ksize = ksize
        self.stride = stride
        self.input_channels = shape[-1]
        self.index = np.zeros(shape)  # mask
        self.output_shape = [shape[0], shape[1] // self.stride,
                             shape[2] // self.stride, self.input_channels]

    def forward(self, x):
        out = np.zeros([x.shape[0], x.shape[1] // self.stride,
                       x.shape[2] // self.stride, self.input_channels])
        for b in range(out.shape[0]):
            for c in range(out.shape[3]):
                for i in range(0, out.shape[1]):
                    for j in range(0, out.shape[2]):
                        out[b, i, j, c] = np.max(x[b, i*self.stride:i*self.stride + self.ksize,
                                                   j*self.stride:j*self.stride + self.ksize, c])
                        index = np.argmax(x[b, i*self.stride:i*self.stride + self.ksize,
                                            j*self.stride:j*self.stride + self.ksize, c])
                        self.index[b, i*self.stride+index//self.stride,
                                   j*self.stride + index % self.stride, c] = 1
        return out

    def forward_gpu(self, x):
        out = np.zeros([x.shape[0], x.shape[1] // self.stride,
                       x.shape[2] // self.stride, self.input_channels])
        grid = (out.shape[0], out.shape[3])
        block = (out.shape[1], out.shape[2])
        pool[grid, block](x, out, self.index)
        cuda.synchronize()
        return out

    def gradient(self, eta):
        tmp = np.repeat(eta, self.stride, axis=1)
        tmp = np.repeat(tmp, self.stride, axis=2)
        return tmp * self.index

    def gradient_gpu(self, eta):
        ret = np.zeros(self.index.shape)
        grid = (eta.shape[0], eta.shape[3])
        block = (eta.shape[1], eta.shape[2])
        pool_bp[grid, block](eta, ret, self.index)
        return ret


@cuda.jit()
def pool(inputs, outputs, mask):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    image = cuda.blockIdx.x
    channel = cuda.blockIdx.y
    outputs[image, tx, ty, channel] = max(inputs[image, 2*tx, 2*ty, channel],
                                          inputs[image, 2*tx, 2*ty+1, channel],
                                          inputs[image, 2*tx+1, 2*ty, channel],
                                          inputs[image, 2*tx+1, 2*ty+1, channel])
    idx = 0
    if outputs[image, tx, ty, channel] == inputs[image, 2*tx, 2*ty, channel]:
        idx = 0
    elif outputs[image, tx, ty, channel] == inputs[image, 2*tx, 2*ty+1, channel]:
        idx = 1
    elif outputs[image, tx, ty, channel] == inputs[image, 2*tx+1, 2*ty, channel]:
        idx = 2
    else:
        idx = 3
    mask[image, tx*2+idx//2, ty*2+idx % 2, channel] = 1


@cuda.jit()
def pool_bp(inputs, outputs, mask):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    image = cuda.blockIdx.x
    channel = cuda.blockIdx.y
    outputs[image, 2*tx, 2*ty, channel] = inputs[image, tx, ty, channel] *mask[image, 2*tx, 2*ty, channel]
    outputs[image, 2*tx, 2*ty+1, channel] = inputs[image, tx, ty, channel] * mask[image, 2*tx, 2*ty+1, channel]
    outputs[image, 2*tx+1, 2*ty, channel] = inputs[image, tx, ty, channel] * mask[image, 2*tx+1, 2*ty, channel]
    outputs[image, 2*tx+1, 2*ty+1, channel] = inputs[image, tx, ty, channel] * mask[image, 2*tx+1, 2*ty+1, channel]

class FullyConnect(object):
    def __init__(self, shape, output_num=216):
        self.input_shape = shape
        self.batchsize = shape[0]
        input_len = reduce(lambda x, y: x * y, shape[1:])
        self.weights = np.random.standard_normal((input_len, output_num))/100
        self.bias = np.random.standard_normal(output_num)/10
        self.output_shape = [self.batchsize, output_num]
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)

        # adam parameter
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.m_t_w = np.zeros(self.weights.shape)
        self.v_t_w = np.zeros(self.weights.shape)
        self.m_t_b = np.zeros(self.bias.shape)
        self.v_t_b = np.zeros(self.bias.shape)
        self.t = 0

    def forward(self, x):
        self.x = x.reshape([self.batchsize, -1])
        output = np.dot(self.x, self.weights)+self.bias
        return output

    def gradient(self, eta):
        # BP
        for i in range(eta.shape[0]):
            col_x = self.x[i][:, np.newaxis]
            eta_i = eta[i][:, np.newaxis].T
            self.w_gradient += np.dot(col_x, eta_i)
            self.b_gradient += eta_i.reshape(self.bias.shape)
        # derivation
        next_eta = np.dot(eta, self.weights.T)
        next_eta = np.reshape(next_eta, self.input_shape)
        return next_eta

    def adam(self, alpha=0.0001, weight_decay=0.0004, batch_size=32):
        # weight_decay = L2 regularization
        self.weights *= (1 - weight_decay)
        self.bias *= (1 - weight_decay)
        self.t += 1
        learning_rate_t = alpha * math.sqrt(
            1 - pow(self.beta2, self.t)) / (1 - pow(self.beta1, self.t))
        self.m_t_w = self.beta1 * self.m_t_w + (
            1 - self.beta1) * self.w_gradient / batch_size
        self.v_t_w = self.beta2 * self.v_t_w + (
            1 - self.beta2) * ((self.w_gradient / batch_size) ** 2)
        self.m_t_b = self.beta1 * self.m_t_b + (
            1 - self.beta1) * self.b_gradient / batch_size
        self.v_t_b = self.beta2 * self.v_t_b + (
            1 - self.beta2) * ((self.b_gradient / batch_size) ** 2)
        self.weights -= learning_rate_t * self.m_t_w / \
            (self.v_t_w + self.epsilon) ** 0.5
        self.bias -= learning_rate_t * self.m_t_b / \
            (self.v_t_b + self.epsilon) ** 0.5
        # zero gradient
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)


class Relu(object):
    def __init__(self, shape):
        self.eta = np.zeros(shape)
        self.x = np.zeros(shape)
        self.output_shape = shape

    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def gradient(self, eta):
        self.eta = eta
        self.eta[self.x < 0] = 0
        return self.eta


class DropOut():
    def __init__(self, dropout_ratio=0.25):
        self.dropout_ratio = dropout_ratio
        self.index = None

    def forward(self, x, training=True):
        if training:
            self.index = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.index
        else:
            return x * (1.0 - self.dropout_ratio)

    def gradient(self, eta):
        return eta * self.index
