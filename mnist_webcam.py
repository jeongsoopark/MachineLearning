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
import cv2
import numpy as np

from PIL import Image, ImageEnhance

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Open camera interface for input
  cv2.namedWindow("hackaday_AP")
  vc = cv2.VideoCapture(0)

  if vc.isOpened():
      ret, frame = vc.read()
  else:
      ret = False

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))

  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  #train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
  train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # Train
  for step in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if(step % 20) == 0:
      print("Cross entropy: ", sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys}))
      print("Accuracy: ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


  prediction = tf.argmax(y, 1)

  # camera input
  camWidth, camHeight, camChannels = frame.shape
  while ret:
      gray_image =  cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      ret, binary_image = cv2.threshold(gray_image, 96, 255, cv2.THRESH_BINARY_INV)
      resized_image = cv2.resize(binary_image, (28, 28), interpolation=cv2.INTER_LINEAR)

      im2 = tf.image.convert_image_dtype(resized_image, tf.float32)
      unrolled = tf.reshape(im2, [-1, 784])
      cv2.imshow("greyscale", binary_image)
      cv2.imshow("resized", resized_image)

      print("Inference: ", (sess.run(prediction, feed_dict={x: unrolled.eval()})))

      ret, frame = vc.read()
      key = cv2.waitKey(20)
      if key == 27:  # exit on ESC
          break

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)