# !/usr/bin/env python3
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: evalute.py 
@Time: 2018/12/3 14:54
@Software: PyCharm 
@Description: 加载训练好的模型进行预测
"""
from data_loader import *
from keras.models import load_model
from keras import backend as K
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

# 数据准备
task = 'ccks'
if task == 'atec':
    train = load_atec()
else:
    train, _, _ = load_ccks()

MAX_LENGTH = select_best_length(train)
datas, word_dict = build_data(train)
# train_w2v(datas)
VOCAB_SIZE = len(word_dict)
embeddings_dict = load_pretrained_embedding()
embedding_matrix = build_embedding_matrix(word_dict, embeddings_dict,
                                          VOCAB_SIZE, EMBEDDING_DIM)
left_x_train, right_x_train, y_train = convert_data(datas, word_dict, MAX_LENGTH)


def exponent_neg_manhattan_distance(sent_left, sent_right):
    '''基于曼哈顿空间距离计算两个字符串语义空间表示相似度计算'''
    return K.exp(-K.sum(K.abs(sent_left - sent_right), axis=1, keepdims=True))

config_obj={'exponent_neg_manhattan_distance':exponent_neg_manhattan_distance}
model=load_model('model/tokenvec_bilstm2_siamese_model.h5',config_obj,compile=False)


y=model.predict(
        x=[left_x_train, right_x_train],
        batch_size=BATCH_SIZE,
    )


print(y)