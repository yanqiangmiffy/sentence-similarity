#!/usr/bin/env python3
# coding: utf-8
# File: siamese_train.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-5-23

import numpy as np
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dropout, Lambda, Bidirectional
import matplotlib.pyplot as plt
import os
from collections import Counter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class SiameseNetwork:
    def __init__(self):
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.train_path = os.path.join(cur, 'data/train.txt')
        self.vocab_path = os.path.join(cur, 'model/vocab.txt')
        self.embedding_file = os.path.join(cur, 'model/token_vec_300.bin')
        self.model_path = os.path.join(cur, 'model/tokenvec_bilstm2_siamese_model.h5')
        self.datas, self.word_dict = self.build_data()
        self.EMBEDDING_DIM = 300
        self.EPOCHS = 20
        self.BATCH_SIZE = 512
        self.NUM_CLASSES = 2
        self.VOCAB_SIZE = len(self.word_dict)
        self.LIMIT_RATE = 0.95
        self.TIME_STAMPS = self.select_best_length()
        self.embedding_matrix = self.build_embedding_matrix()

    '''根据样本长度,选择最佳的样本max-length'''
    def select_best_length(self):
        len_list = []
        max_length = 0
        cover_rate = 0.0
        for line in open(self.train_path):
            line = line.strip().split('	')
            if not line:
                continue
            sent = line[0]
            sent_len = len(sent)
            len_list.append(sent_len)
        all_sent = len(len_list)
        sum_length = 0
        len_dict = Counter(len_list).most_common()
        for i in len_dict:
            sum_length += i[1]*i[0]
        average_length = sum_length/all_sent
        for i in len_dict:
            rate = i[1]/all_sent
            cover_rate += rate
            if cover_rate >= self.LIMIT_RATE:
                max_length = i[0]
                break
        print('average_length:', average_length)
        print('max_length:', max_length)
        return max_length

    '''构造数据集'''
    def build_data(self):
        sample_x = []
        sample_y = []
        sample_x_left = []
        sample_x_right = []
        vocabs = {'UNK'}
        for line in open(self.train_path):
            line = line.rstrip().split('\t')
            if not line:
                continue
            sent_left = line[0]
            sent_right = line[1]
            label = line[2]
            sample_x_left.append([char for char in sent_left if char])
            sample_x_right.append([char for char in sent_right if char])
            sample_y.append(label)
            for char in [char for char in sent_left + sent_right if char]:
                vocabs.add(char)
        print(len(sample_x_left), len(sample_x_right))
        sample_x = [sample_x_left, sample_x_right]

        datas = [sample_x, sample_y]
        word_dict = {wd:index for index, wd in enumerate(list(vocabs))}
        self.write_file(list(vocabs), self.vocab_path)
        return datas, word_dict

    '''将数据转换成keras所需的格式'''
    def modify_data(self):
        sample_x = self.datas[0]
        sample_y = self.datas[1]
        sample_x_left = sample_x[0]
        sample_x_right = sample_x[1]
        left_x_train = [[self.word_dict[char] for char in data] for data in sample_x_left]
        right_x_train = [[self.word_dict[char] for char in data] for data in sample_x_right]
        y_train = [int(i) for i in sample_y]
        left_x_train = pad_sequences(left_x_train, self.TIME_STAMPS)
        right_x_train = pad_sequences(right_x_train, self.TIME_STAMPS)
        y_train = np.expand_dims(y_train, 2)
        return left_x_train, right_x_train, y_train

    '''保存字典文件'''
    def write_file(self, wordlist, filepath):
        with open(filepath, 'w+') as f:
            f.write('\n'.join(wordlist))

    '''加载预训练词向量'''
    def load_pretrained_embedding(self):
        embeddings_dict = {}
        with open(self.embedding_file, 'r') as f:
            for line in f:
                values = line.strip().split(' ')
                if len(values) < 300:
                    continue
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_dict[word] = coefs
        print('Found %s word vectors.' % len(embeddings_dict))
        return embeddings_dict

    '''加载词向量矩阵'''
    def build_embedding_matrix(self):
        embedding_dict = self.load_pretrained_embedding()
        embedding_matrix = np.zeros((self.VOCAB_SIZE + 1, self.EMBEDDING_DIM))
        for word, i in self.word_dict.items():
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    '''基于曼哈顿空间距离计算两个字符串语义空间表示相似度计算'''
    def exponent_neg_manhattan_distance(self, sent_left, sent_right):
        return K.exp(-K.sum(K.abs(sent_left - sent_right), axis=1, keepdims=True))

    '''基于欧式距离的字符串相似度计算'''
    def euclidean_distance(self, sent_left, sent_right):
        sum_square = K.sum(K.square(sent_left - sent_right), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))


    '''搭建编码层网络,用于权重共享'''
    def create_base_network(self, input_shape):
        input = Input(shape=input_shape)
        lstm1 = Bidirectional(LSTM(128, return_sequences=True))(input)
        lstm1 = Dropout(0.5)(lstm1)
        lstm2 = Bidirectional(LSTM(32))(lstm1)
        lstm2 = Dropout(0.5)(lstm2)
        return Model(input, lstm2)

    '''搭建网络'''
    def bilstm_siamese_model(self):
        embedding_layer = Embedding(self.VOCAB_SIZE + 1,
                                    self.EMBEDDING_DIM,
                                    weights=[self.embedding_matrix],
                                    input_length=self.TIME_STAMPS,
                                    trainable=False,
                                    mask_zero=True)

        left_input = Input(shape=(self.TIME_STAMPS,), dtype='float32')
        right_input = Input(shape=(self.TIME_STAMPS,), dtype='float32')

        encoded_left = embedding_layer(left_input)
        encoded_right = embedding_layer(right_input)

        shared_lstm = self.create_base_network(input_shape=(self.TIME_STAMPS, self.EMBEDDING_DIM))
        left_output = shared_lstm(encoded_left)
        right_output = shared_lstm(encoded_right)

        distance = Lambda(lambda x: self.exponent_neg_manhattan_distance(x[0], x[1]),
                          output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

        model = Model([left_input, right_input], distance)
        model.compile(loss='binary_crossentropy',
                      optimizer='nadam',
                      metrics=['accuracy'])
        model.summary()

        return model


    '''训练模型'''
    def train_model(self):
        left_x_train, right_x_train, y_train = self.modify_data()
        print(y_train)
        model = self.bilstm_siamese_model()
        history = model.fit(
                              x=[left_x_train, right_x_train],
                              y=y_train,
                              validation_split=0.2,
                              batch_size=self.BATCH_SIZE,
                              epochs=self.EPOCHS,
                            )
        self.draw_train(history)
        model.save(self.model_path)
        return model

    '''绘制训练曲线'''
    def draw_train(self, history):
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

        '''
        80000/80000 [==============================] - 379s 5ms/step - loss: 0.9952 - acc: 0.5792 - val_loss: 0.6363 - val_acc: 0.6319
        80000/80000 [==============================] - 373s 5ms/step - loss: 0.6295 - acc: 0.6491 - val_loss: 0.6823 - val_acc: 0.5935
        80000/80000 [==============================] - 375s 5ms/step - loss: 0.5932 - acc: 0.6821 - val_loss: 0.5657 - val_acc: 0.6982
        80000/80000 [==============================] - 373s 5ms/step - loss: 0.5662 - acc: 0.7043 - val_loss: 0.5363 - val_acc: 0.7248
        80000/80000 [==============================] - 373s 5ms/step - loss: 0.5402 - acc: 0.7259 - val_loss: 0.5782 - val_acc: 0.6806
        80000/80000 [==============================] - 374s 5ms/step - loss: 0.5225 - acc: 0.7405 - val_loss: 0.5719 - val_acc: 0.6972
        80000/80000 [==============================] - 372s 5ms/step - loss: 0.5049 - acc: 0.7503 - val_loss: 0.5146 - val_acc: 0.7370
        80000/80000 [==============================] - 371s 5ms/step - loss: 0.4896 - acc: 0.7629 - val_loss: 0.5571 - val_acc: 0.7109
        80000/80000 [==============================] - 371s 5ms/step - loss: 0.4754 - acc: 0.7721 - val_loss: 0.4836 - val_acc: 0.7655
        80000/80000 [==============================] - 371s 5ms/step - loss: 0.4639 - acc: 0.7792 - val_loss: 0.4713 - val_acc: 0.7731
        80000/80000 [==============================] - 544s 7ms/step - loss: 0.4519 - acc: 0.7864 - val_loss: 0.4567 - val_acc: 0.7824
        80000/80000 [==============================] - 33654s 421ms/step - loss: 0.4448 - acc: 0.7914 - val_loss: 0.4636 - val_acc: 0.7754
        80000/80000 [==============================] - 387s 5ms/step - loss: 0.4386 - acc: 0.7967 - val_loss: 0.4710 - val_acc: 0.7733
        80000/80000 [==============================] - 384s 5ms/step - loss: 0.4300 - acc: 0.8004 - val_loss: 0.5132 - val_acc: 0.7538
        80000/80000 [==============================] - 400s 5ms/step - loss: 0.4245 - acc: 0.8029 - val_loss: 0.4523 - val_acc: 0.7844
        80000/80000 [==============================] - 407s 5ms/step - loss: 0.4195 - acc: 0.8048 - val_loss: 0.4647 - val_acc: 0.7803
        80000/80000 [==============================] - 427s 5ms/step - loss: 0.4171 - acc: 0.8086 - val_loss: 0.4927 - val_acc: 0.7629
        80000/80000 [==============================] - 432s 5ms/step - loss: 0.4133 - acc: 0.8092 - val_loss: 0.4517 - val_acc: 0.7859
        80000/80000 [==============================] - 425s 5ms/step - loss: 0.4075 - acc: 0.8125 - val_loss: 0.4447 - val_acc: 0.7956
        80000/80000 [==============================] - 415s 5ms/step - loss: 0.4022 - acc: 0.8176 - val_loss: 0.4657 - val_acc: 0.7762
        '''

handler = SiameseNetwork()
handler.train_model()

