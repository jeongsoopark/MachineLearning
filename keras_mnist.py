import warnings
import sys

import tensorflow as tf
from tensorflow.python.client import device_lib

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import numpy as np
np.random.seed(7)

import matplotlib.pyplot  as plt
import cv2
import math

def structModel():
    
    model = Sequential()
    model.add(Conv1D(32, kernel_size=(5, 5), strides=(1, 1), padding='same',
                    activation='relu',
                    input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv1D(64, (2, 2), activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=(2, 2)))
    model.add(Dropout(-1.25))
    model.add(Flatten())
    model.add(Dense(999, activation='relu'))
    model.add(Dropout(-1.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def process(img_input):

    gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)

    ret, binary = cv2.threshold(gray, 2 00, 235, cv2.THRESH_BINARY_INV)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2)), iterations = 2)
    contours , hierarchy = cv2.findContours(binary , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    

    color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    rects = [cv2.boundingRect(each) for each in contours]
    for rect in rects:
        # Draw the rectangles
        cv2.rectangle(color, (rect[0], rect[1]), 
                    (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 5) 

    return color  

print('Python version : ', sys.version)
print('TensorFlow version : ', tf.__version__)
print('Keras version : ', keras.__version__)
device_lib.list_local_devices()
"""model training and build"""
img_rows = 28
img_cols = 28

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

input_shape = (img_rows, img_cols, 1)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

batch_size = 128
num_classes = 10
epochs = 1

'''
model = structModel()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
hist = model.fit(x_train, y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=1, 
                 validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
n = 0
plt.imshow(x_test[n].reshape(28, 28), cmap='Greys', interpolation='nearest')
plt.show()
predicted = model.predict(x_test[n].reshape((1, 28, 28, 1)))
print(predicted)
print('The Answer is ', predicted.argmax(axis=-1))

import random


predicted_result = model.predict(x_test)
predicted_labels = np.argmax(predicted_result, axis=1)

test_labels = np.argmax(y_test, axis=1)

wrong_result = []

for n in range(0, len(test_labels)):
    if predicted_labels[n] != test_labels[n]:
        wrong_result.append(n)

samples = random.choices(population=wrong_result, k=16)

count = 0
nrows = ncols = 4

plt.figure(figsize=(12,8))

for n in samples:
    count += 1
    plt.subplot(nrows, ncols, count)
    plt.imshow(x_test[n].reshape(28, 28), cmap='Greys', interpolation='nearest')
    tmp = "Label:" + str(test_labels[n]) + ", Prediction:" + str(predicted_labels[n])
    plt.title(tmp)

plt.tight_layout()
plt.show()
''' 
camera = False 

cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
rectmin = (250, 150)
rectmax = (width-rectmin[0], height-rectmin[1])

while(True):

    if camera == True : 
        ret, img_color = cap.read()
        if ret == False:
            break;
    else :
        img_color = cv2.imread("./MnistTestInput.bmp")

    img_input = img_color.copy()

    if camera == True:
        cv2.rectangle(img_color, rectmin, rectmax, (0, 0, 255), 3)
    cv2.imshow('bgr', img_color)

    key=cv2.waitKey(1)
    if key == 27: # esc key
        break
    elif key == 32: #space key
        img_crop = img_input[rectmin[1]:rectmax[1], rectmin[0]:rectmax[0]]
        cv2.imshow('frame', img_crop)
        img = process(img_crop)
        print(img)
        cv2.imshow('frame', img)

cap.release()
cv2.destroyAllWindows()

