# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 17:35:30 2022

@author: Shaochang Liu
"""
import numpy as np
from layers_numpy import Conv2D, FullyConnect, MaxPooling, Relu
from loss_numpy import Softmax

import time
import struct
from glob import glob


def load_mnist(path, count, kind='train'):
    """Load MNIST data from `path`"""
    images_path = glob('./%s/%s*3-ubyte' % (path, kind))[0]
    labels_path = glob('./%s/%s*1-ubyte' % (path, kind))[0]
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    labels = labels[0:count]
    images = images[0:count]
    return images, labels


images, labels = load_mnist('./mnist', 9600)
test_images, test_labels = load_mnist('./mnist', 960, 't10k')

batch_size = 32
learning_rate = 1e-3

conv1 = Conv2D([batch_size, 28, 28, 1], 12)
relu1 = Relu(conv1.output_shape)
pool1 = MaxPooling(relu1.output_shape)
conv2 = Conv2D(pool1.output_shape, 24)
relu2 = Relu(conv2.output_shape)
pool2 = MaxPooling(relu2.output_shape)
fc = FullyConnect(pool2.output_shape, 10)
sf = Softmax(fc.output_shape)

print("training start time: ", time.strftime(
    "%Y-%m-%d %H:%M:%S", time.localtime()))

for epoch in range(1):
    val_acc = 0

    # train
    for i in range(images.shape[0] // batch_size):
        batch_loss = 0

        img = images[i * batch_size:(i + 1) *
                     batch_size].reshape([batch_size, 28, 28, 1])
        label = labels[i * batch_size:(i + 1) * batch_size]

        conv1_out = relu1.forward(conv1.forward(img))
        pool1_out = pool1.forward(conv1_out)
        conv2_out = relu2.forward(conv2.forward(pool1_out))
        pool2_out = pool2.forward(conv2_out)
        fc_out = fc.forward(pool2_out)

        batch_loss += sf.cal_loss(fc_out, np.array(label))
        sf.gradient()
        conv1.gradient(relu1.gradient(pool1.gradient(
            conv2.gradient(relu2.gradient(pool2.gradient(
                fc.gradient(sf.eta)))))))

        fc.adam(alpha=learning_rate, weight_decay=0.0003)
        conv2.adam(alpha=learning_rate, weight_decay=0.0003)
        conv1.adam(alpha=learning_rate, weight_decay=0.0003)

        if (i+1) % 50 == 0:
            print(time.strftime("%Y-%m-%d %H:%M:%S, ", time.localtime()) +
                  "epoch: %d, batch: %d, avg batch loss: %.4f" % (epoch+1, i+1,
                                                                  batch_loss / batch_size))

    # validation
    for i in range(test_images.shape[0] // batch_size):
        img = test_images[i * batch_size:(i + 1) *
                          batch_size].reshape([batch_size, 28, 28, 1])
        label = test_labels[i * batch_size:(i + 1) * batch_size]
        conv1_out = relu1.forward(conv1.forward(img))
        pool1_out = pool1.forward(conv1_out)
        conv2_out = relu2.forward(conv2.forward(pool1_out))
        pool2_out = pool2.forward(conv2_out)
        fc_out = fc.forward(pool2_out)

        sf.cal_loss(fc_out, np.array(label))

        for j in range(batch_size):
            if np.argmax(sf.softmax[j]) == label[j]:
                val_acc += 1

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) +
          "  epoch: %d , val acc: %.4f" % (epoch+1, val_acc / float(test_images.shape[0])))
