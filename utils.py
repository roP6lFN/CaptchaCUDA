# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 01:27:06 2022

@author: Shaochang Liu
"""

import os
import numpy as np
import random
from PIL import Image


# components
NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
UP_CASE = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
CAPTCHA_LIST = NUMBER + UP_CASE
CAPTCHA_LEN = 4
CAPTCHA_HEIGHT = 24
CAPTCHA_WIDTH = 80


# produce captcha text
def random_captcha_text(char_set=CAPTCHA_LIST, captcha_size=CAPTCHA_LEN):
    captcha_text = [random.choice(char_set) for _ in range(captcha_size)]
    return ''.join(captcha_text)


# RPG -> grey-scale
def convert2gray(img):
    if len(img.shape) > 2:
        img = np.mean(img, -1)
    return img


# produce captcha image
def gen_captcha_text_and_image(data_path):
    all_image = os.listdir(data_path)
    random_file = random.randint(0, len(all_image) - 1)
    name = all_image[random_file][0: CAPTCHA_LEN]
    image = Image.open(os.path.join(data_path, all_image[random_file]))
    image = np.array(image)[1:25, :]
    image = convert2gray(image)
    return name, image


# text -> vector(36*6)
def text2vec(text, captcha_len=CAPTCHA_LEN, captcha_list=CAPTCHA_LIST):
    text_len = len(text)
    if text_len > captcha_len:
        raise ValueError('text length exceed')
    vector = np.zeros(captcha_len * len(captcha_list))
    for i in range(text_len):
        vector[captcha_list.index(text[i])+i*len(captcha_list)] = 1
    return vector


# vector(36) -> text
def vec2text(vec, captcha_list=CAPTCHA_LIST, size=CAPTCHA_LEN):
    vec_idx = vec
    text_list = [captcha_list[v] for v in vec_idx]
    return ''.join(text_list)


# auto generate training dataset
def next_batch(data_path, batch_count=32,
               width=CAPTCHA_WIDTH, height=CAPTCHA_HEIGHT):
    batch_x = np.zeros([batch_count, width * height])
    batch_y = np.zeros([batch_count, CAPTCHA_LEN * len(CAPTCHA_LIST)])
    for i in range(batch_count):
        text, image = gen_captcha_text_and_image(data_path)
        image = convert2gray(image)
        batch_x[i, :] = image.flatten() / 255 # normalize after flatten
        batch_y[i, :] = text2vec(text) # label
    return batch_x, batch_y


if __name__ == '__main__':
    x, y = next_batch('train_data/')
    x = x.reshape([32, CAPTCHA_WIDTH, CAPTCHA_HEIGHT, 1])
    print(x, '\n\n', y)
