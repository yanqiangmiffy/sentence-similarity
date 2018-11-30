# !/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: build_input.py 
@Time: 2018/11/30 17:41
@Software: PyCharm 
@Description: 构建模型的输入
"""
from data_loader import load_atec
from collections import Counter
train=load_atec()


def select_best_length(limit_ratio=0.95):
    """
    根据数据集的句子长度，选择最佳的样本max-length
    :param limit_ratio:句子长度覆盖度，默认覆盖95%以上的句子
    :return:
    """
    len_list = []
    max_length = 0
    cover_rate = 0.0
    for q1,q2 in zip(train['q1'],train['q2']):
        len_list.append(len(q1))
        len_list.append(len(q2))
    all_sent = len(len_list)
    sum_length = 0
    len_dict = Counter(len_list).most_common()
    for i in len_dict:
        sum_length += i[1] * i[0]
    average_length = sum_length / all_sent
    for i in len_dict:
        rate = i[1] / all_sent
        cover_rate += rate
        if cover_rate >= limit_ratio:
            max_length = i[0]
            break
    print('average_length:', average_length)
    print('max_length:', max_length)
    return max_length


def build_data():
    """
    构建数据集
    :return:
    """
    sample_x_left = train.q1.apply(lambda x:[char for char in x if char]).tolist()
    sample_x_right = train.q1.apply(lambda x:[char for char in x if char]).tolist()
    vocabs = {'UNK'}
    for x_left,x_right in zip(sample_x_left,sample_x_right):
        for char in x_left+x_right:
            vocabs.add(char)

    sample_x = [sample_x_left, sample_x_right]
    print(sample_x[0][0])
    sample_y = train.label.tolist()
    print(len(sample_x_left), len(sample_x_right))
    datas = [sample_x, sample_y]
    word_dict = {wd: index for index, wd in enumerate(list(vocabs))}
    vocab_path='model/vocab.txt'
    with open(vocab_path,'w',encoding='utf-8') as f:
        f.write('\n'.join(list(vocabs)))
    return datas, word_dict
build_data()