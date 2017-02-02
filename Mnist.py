# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
#import cv2
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None
learning_rate = 0.1
training_epochs = 15
batch_size = 100
display_step = 1
FLAGS = None

# Import data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

b = tf.Variable(tf.zeros([10]))
W = tf.Variable(tf.zeros([784, 10]))

more_layer = 1

if(more_layer == 1):
    W1 = tf.Variable(tf.zeros([784, 256]))
    W2 = tf.Variable(tf.zeros([256, 256]))
    W3 = tf.Variable(tf.zeros([256, 10]))

    B1 = tf.Variable(tf.zeros([256]))
    B2 = tf.Variable(tf.zeros([256]))
    B3 = tf.Variable(tf.zeros([10]))

    Y1 = tf.nn.relu(tf.matmul(x, W1) + B1)
    Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
    Y3 = tf.matmul(Y2, W3) + B3
    hypothesis = tf.nn.softmax(Y3)
else:
    hypothesis = tf.nn.softmax(tf.matmul(x, W) +b)

# Define cost and optimizer
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#initializing variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    #training
    for epoch in range(training_epochs):
        avg_cost = 0.0
        total_batch = int(mnist.train.num_examples/batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            #fit training using batch data
            sess.run(optimizer, feed_dict={x:batch_xs, y:batch_ys})

            #calculate avg cost
            avg_cost += sess.run(cost, feed_dict={x:batch_xs, y:batch_ys}) / total_batch

        if(epoch % display_step) == 0:
            print(avg_cost)

    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy: ", sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels}))

