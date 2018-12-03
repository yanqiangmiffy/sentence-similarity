# !/usr/bin/env python3
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: evalute.py 
@Time: 2018/12/3 14:54
@Software: PyCharm 
@Description: 加载训练好的模型进行预测
"""

from keras.models import load_model
from keras import backend as K

def exponent_neg_manhattan_distance(sent_left, sent_right):
    '''基于曼哈顿空间距离计算两个字符串语义空间表示相似度计算'''
    return K.exp(-K.sum(K.abs(sent_left - sent_right), axis=1, keepdims=True))

model=load_model('model/tokenvec_bilstm2_siamese_model.h5')