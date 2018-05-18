import csv
import cv2
import os, glob
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import time
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from matplotlib import pyplot as plt
from keras.utils import plot_model

model.save('G:\\pytext\\reco\\model\\my_model.h5')

def load_data():
    train=open('G:\\pytext\\reco\\datasets\\train.txt')
    mape={}
    labels = []
    images = []
    for line in train.readlines():
        k=line.strip().split()
        mape[k[0]]=k[1]
    for root,dirs,files in os.walk('G:\\pytext\\reco\\datasets\\train'):
        for file in files:
            file_path=os.path.join(root,file)
            images.append(skimage.data.imread(file_path))
            labels.append(int(mape[file]))
    return images, labels
images, labels=load_data()




#交叉验证
def cross_validate(Xs, ys):
    X_train, X_test, y_train, y_test = train_test_split(
            Xs, ys, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = cross_validate(data, label)

#X是Images Y是Labels
# normalize inputs from 0-255 and 0.0-1.0
X_train = np.array(X_train).astype('float32')
X_test = np.array(X_test).astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode outputs 独热编码
y_train = np.array(y_train)
y_test = np.array(y_test)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
print("Data normalized and hot encoded.")

np.save("G:\\pytext\\reco\\datasets\\X_train.npy",X_train)
np.save("G:\\pytext\\reco\\datasets\\X_test.npy",X_test)
np.save("G:\\pytext\\reco\\datasets\\y_train.npy",y_train)
np.save("G:\\pytext\\reco\\datasets\\y_test.npy",y_test)

X_train = np.load("G:\\pytext\\reco\\datasets\\X_train.npy")
X_test = np.load("G:\\pytext\\reco\\datasets\\X_test.npy")
y_train = np.load("G:\\pytext\\reco\\datasets\\y_train.npy")
y_test = np.load("G:\\pytext\\reco\\datasets\\y_test.npy")


def createCNNModel(num_classes):
    """ Adapted from: # http://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/
# """
    # Create the model
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(125, 200, 3), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    epochs = 25  # >>> should be 25+
    lrate = 0.01
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print(model.summary())
    return model, epochs

# create our CNN model
model, epochs = createCNNModel(num_classes)



# fit and run our model
seed = 7
np.random.seed(seed)
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=epochs, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

print("done")



test_data = np.array(test_data).astype('float32')
test_data = test_data / 255.0


cnn_prediction = model.predict_proba(test_data)

prediction=[]
for i in cnn_prediction:
    min=-1.0
    pos=0
    for k in range(len(i)):
        if i[k]>min:
            min=i[k]
            pos=k
    prediction.append(pos)

def write_test_labels():
    fr=open('G:\\pytext\\reco\\testcnn1.csv','w')
    i=0
    for root,dirs,files in os.walk('G:\\pytext\\reco\\datasets\\test'):
        for file in files:
            fr.write(file)
            fr.write(' ')
            fr.write(str(prediction[int(mape[file])]))
            fr.write("\n")
            i=i+1
    fr.close()










