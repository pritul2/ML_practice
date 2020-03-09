'''
This code is made to learn how LeNet works. 
Dataset used is the MNIST handwritten digits.
Accuracy obtained at testing is 99%
'''
import keras
from keras.layers import Conv2D,AveragePooling2D,Flatten,Dense
from keras.models import Sequential
from keras.utils import np_utils
from keras.datasets import mnist
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_model():
    model = Sequential()
    #LeNet is taking only 32x32x3 image
    model.add(Conv2D(6,(3,3),activation="relu",input_shape=(32,32,1))) #n-f+1 = (30,30,6)
    model.add(AveragePooling2D()) #(15,15,6)
    model.add(Conv2D(16,(3,3),activation="relu")) #15-3+1 = (13,13,16)
    model.add(AveragePooling2D()) #(6,6,16)
    model.add(Flatten()) #(576)
    model.add(Dense(units=120, activation='relu')) 
    model.add(Dense(units=84, activation='relu'))
    model.add(Dense(units=10, activation = 'softmax')) #MNIST has 10 class#
    print(model.summary())
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    return model

EPOCHS = 10
BATCH_SIZE = 128

((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()
trainData =  np.pad(trainData,((0,0),(2,2),(2,2)))
testData = np.pad(testData,((0,0),(2,2),(2,2)))
trainData = trainData.reshape((trainData.shape[0],  32, 32,1))

testData = testData.reshape((testData.shape[0], 32, 32,1))
# scale data to the range of [0, 1]
trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0

trainLabels = np_utils.to_categorical(trainLabels, 10)
testLabels = np_utils.to_categorical(testLabels, 10)
model = load_model()
model.fit(trainData, trainLabels, batch_size=128, epochs=20,verbose=1)

(loss, accuracy) = model.evaluate(testData, testLabels,batch_size=128, verbose=1)
print("accuracy is ",accuracy*100) #99%#

img = cv2.imread("/Users/prituldave/Downloads/test_img.png",0)
img = cv2.resize(img,(32,32))
probs = model.predict(img[np.newaxis,...,np.newaxis])
prediction = probs.argmax(axis=1)
print(prediction)
