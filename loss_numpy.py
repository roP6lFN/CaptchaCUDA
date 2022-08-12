# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 17:35:30 2022

@author: Shaochang Liu
"""
import numpy as np

class Softmax(object):
    def __init__(self, shape):
        self.softmax = np.zeros(shape)
        self.eta = np.zeros(shape)
        self.batchsize = shape[0]

    def cal_loss(self, prediction, label):
        self.label = label
        self.prediction = prediction
        self.predict(prediction)
        self.loss = 0
        for i in range(self.batchsize):
            self.loss += np.log(np.sum(np.exp(prediction[i]))
                                ) - prediction[i, label[i]]
        return self.loss

    def predict(self, prediction): # probability distribution
        exp_prediction = np.zeros(prediction.shape)
        self.softmax = np.zeros(prediction.shape)
        for i in range(self.batchsize):
            prediction[i, :] -= np.max(prediction[i, :])
            exp_prediction[i] = np.exp(prediction[i])
            self.softmax[i] = exp_prediction[i]/np.sum(exp_prediction[i])
        return self.softmax

    def gradient(self):
        self.eta = self.softmax.copy()
        for i in range(self.batchsize):
            self.eta[i, self.label[i]] -= 1
        return self.eta


class Sigmoid(object):
    def __init__(self): 
        pass

    def loss(self, y, p):
        p = np.clip(p, 1e-5, 1 - 1e-5)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def gradient(self, y, p):
        p = np.clip(p, 1e-5, 1 - 1e-5)
        return - (y / p) + (1 - y) / (1 - p)
