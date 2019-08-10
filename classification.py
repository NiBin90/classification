#! -*- coding:utf-8 -*-

import pdb
from keras.optimizers import Adam
import keras.backend as K
from keras.models import Model
from keras.layers import *
import json
import numpy as np
import pandas as pd
from random import choice
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import re
import os
import codecs
import keras as K

maxlen = 100
config_path = '../bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../bert/chinese_L-12_H-768_A-12/vocab.txt'


token_dict = {}

with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

tokenizer = OurTokenizer(token_dict)

train_path_file = "./data/D.train"
test_path_file = "./data/D.test"
NUM_CLASS = 9
with open(train_path_file, "r", encoding="utf8") as f:
    train_data_all = f.read().splitlines()
with open(test_path_file, "r", encoding="utf8") as f:
    valid_data_all = f.read().splitlines()
train_data = []
valid_data = []
for i in train_data_all:
    train_data.append((seq_padding(i.split("\t")[1]), i.split("\t")[0]))
for j in valid_data_all:
    valid_data.append((seq_padding(j.split("\t")[1]), j.split("\t")[0]))
data = []


class data_generator:
    def __init__(self, data, batch_size=12):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = range(len(self.data))

            # np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]
                x1, x2 = tokenizer.encode(first=text)
                y = d[1]
                y = to_categorical(y, num_classes=NUM_CLASS)
                X1.append(x1)
                X2.append(x2)
                Y.append(y)
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []

# 取后面四层 output_layer_num = 4
bert_model = load_trained_model_from_checkpoint(
    config_path, checkpoint_path, seq_len=None, output_layer_num=4)

for l in bert_model.layers:
    l.trainable = True
# import pdb; pdb.set_trace()
x1_in = Input(shape=(maxlen+2,))
x2_in = Input(shape=(maxlen+2,))

x = bert_model([x1_in, x2_in])
# import pdb; pdb.set_trace()
# 
x = Lambda(lambda x: x[:, :])(x)
# x = Lambda(lambda x: K.backend.expand_dims(x, axis=-1))(x)
# x = Dense(1024, activation='relu')(x)
# x = Dense(768, activation='relu')(x)
# import pdb; pdb.set_trace()
x = Lambda(lambda x: K.backend.expand_dims(x, axis=-1))(x)
# x = Lambda(lambda x: K.backend.expand_dims(x, axis=-1))(x)
# x = Convolution2D(strides=1, filters=32, kernel_size=(3072, 1))(x)
x = Convolution2D(strides=1, filters=16, kernel_size=(5, 1), activation='relu')(x)
x = Convolution2D(strides=1, filters=8, kernel_size=(5, 1), activation='relu')(x)
x = AveragePooling2D()(x)
x = SpatialDropout2D(0.5)(x)
x = Flatten()(x)
x = Dropout(rate=0.5)(x)
p = Dense(NUM_CLASS, activation='softmax')(x)

model = Model([x1_in, x2_in], p)
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(1e-5),  # 用足够小的学习率
    metrics=['accuracy']
)
model.summary()


train_D = data_generator(train_data)
valid_D = data_generator(valid_data)
# pdb.set_trace()
model.fit_generator(
    train_D.__iter__(),
    steps_per_epoch=len(train_D),
    epochs=5,
    validation_data=valid_D.__iter__(),
    validation_steps=len(valid_D)
)
