#coding:utf8
import re
import cv2
import os
import numpy as np
import cv2.cv as cv
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Flatten
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.layers.advanced_activations import LeakyReLU

from keras.utils import np_utils




#构建卷积神经网络
def cnn_model(train_data,train_label,test_data,test_label):
    model = Sequential()
    model.add(Convolution2D(
        nb_filter = 12,
        nb_row = 3,
        nb_col = 3,
        border_mode = 'valid',
        dim_ordering = 'th',
        input_shape = (1,128,192)))
    model.add(Activation('relu'))#激活函数使用修正线性单元
    model.add(MaxPooling2D(
        pool_size = (2,2),
        strides = (2,2),
        border_mode = 'valid'))
    model.add(Convolution2D(
        24,
        3,
        3,
        border_mode = 'valid',
        dim_ordering = 'th'))
    model.add(Activation('relu'))
#池化层 24×29×29
    model.add(MaxPooling2D(
        pool_size = (2,2),
        strides = (2,2),
        border_mode = 'valid'))
    model.add(Convolution2D(
        48,
        3,
        3,
        border_mode = 'valid',
        dim_ordering = 'th'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(
        pool_size = (2,2),
        strides =(2,2),
        border_mode = 'valid'))
    model.add(Flatten())
    model.add(Dense(20))
    model.add(Activation(LeakyReLU(0.3)))
    #model.add(Dropout(0.5))
    model.add(Dense(20))
    model.add(Activation(LeakyReLU(0.3)))
    #model.add(Dropout(0.4))
    model.add(Dense(5,init = 'normal'))
    model.add(Activation('softmax'))
    adam = Adam(lr = 0.001)
    model.compile(optimizer = adam,
            loss =  'categorical_crossentropy',
            metrics = ['accuracy'])
    print('----------------training-----------------------')
    model.fit(train_data,train_label,batch_size = 12,nb_epoch = 35,shuffle = True,show_accuracy = True,validation_split = 0.1)
    print('----------------testing------------------------')
    loss,accuracy = model.evaluate(test_data,test_label)
    print('\n test loss:',loss)
    print('\n test accuracy',accuracy)




cnn_model(train_data,train_label,test_data,test_label)