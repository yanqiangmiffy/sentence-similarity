# -*- coding: utf-8 -*-
# @Time    : 2018/11/30 22:44
# @Author  : quincyqiang
# @File    : train_siamese_network.py
# @Software: PyCharm

import numpy as np
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dropout, Lambda, Bidirectional
import matplotlib.pyplot as plt
import os
from collections import Counter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from build_input import *

# 参数设置
BATCH_SIZE = 512
EMBEDDING_DIM = 300
EPOCHS = 20
model_path = 'model/tokenvec_bilstm2_siamese_model.h5'
# 数据准备

MAX_LENGTH = select_best_length()
datas, word_dict = build_data()
VOCAB_SIZE = len(word_dict)
left_x_train, right_x_train, y_train = convert_data(datas, word_dict, MAX_LENGTH)
embeddings_dict = load_pretrained_embedding()
embedding_matrix = build_embedding_matrix(word_dict, embeddings_dict,
                                          VOCAB_SIZE, EMBEDDING_DIM)


def create_base_network(input_shape):
    '''搭建编码层网络,用于权重共享'''
    input = Input(shape=input_shape)
    lstm1 = Bidirectional(LSTM(128, return_sequences=True))(input)
    lstm1 = Dropout(0.5)(lstm1)
    lstm2 = Bidirectional(LSTM(32))(lstm1)
    lstm2 = Dropout(0.5)(lstm2)
    return Model(input, lstm2)


def exponent_neg_manhattan_distance(sent_left, sent_right):
    '''基于曼哈顿空间距离计算两个字符串语义空间表示相似度计算'''
    return K.exp(-K.sum(K.abs(sent_left - sent_right), axis=1, keepdims=True))


def bilstm_siamese_model():
    '''搭建网络'''
    embedding_layer = Embedding(VOCAB_SIZE + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_LENGTH,
                                trainable=False,
                                mask_zero=True)
    left_input = Input(shape=(MAX_LENGTH,), dtype='float32')
    right_input = Input(shape=(MAX_LENGTH,), dtype='float32')
    encoded_left = embedding_layer(left_input)
    encoded_right = embedding_layer(right_input)
    shared_lstm = create_base_network(input_shape=(MAX_LENGTH, EMBEDDING_DIM))
    left_output = shared_lstm(encoded_left)
    right_output = shared_lstm(encoded_right)
    distance = Lambda(lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
                      output_shape=lambda x: (x[0][0], 1))([left_output, right_output])
    model = Model([left_input, right_input], distance)
    model.compile(loss='binary_crossentropy',
                  optimizer='nadam',
                  metrics=['accuracy'])
    model.summary()
    return model


def draw_train(history):
    '''绘制训练曲线'''
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def train_model():
    '''训练模型'''
    model = bilstm_siamese_model()
    history = model.fit(
        x=[left_x_train, right_x_train],
        y=y_train,
        validation_split=0.2,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
    )
    draw_train(history)
    model.save(model_path)
    return model


train_model()