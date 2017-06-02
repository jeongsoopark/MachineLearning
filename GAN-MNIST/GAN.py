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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

FLAGS = None


def xavier_init(size):
  in_dim = size[0]
  xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
  return tf.random_normal(shape=size, stddev=xavier_stddev)


# Discriminator Net
X = tf.placeholder(tf.float32, shape=[None, 784], name='X')

D_W1 = tf.Variable(xavier_init([784, 128]))
D_B1 = tf.Variable(tf.zeros(shape=[128]), name='D_1')

D_W2 = tf.Variable(xavier_init([128, 1]))
D_B2 = tf.Variable(tf.zeros(shape=[1]), name='D_b2')

theta_D = [D_W1, D_W2, D_B1, D_B2]

# Generator Net
Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')

G_W1 = tf.Variable(xavier_init([100, 128]))
G_B1 = tf.Variable(tf.zeros(shape=[128]), name='G_B1')

G_W2 = tf.Variable(xavier_init([128, 784]))
G_B2 = tf.Variable(tf.zeros(shape=[784]), name='G_B2')

theta_G = [G_W1, G_W2, G_B1, G_B2]

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


def sample_Z(m, n):
  return np.random.uniform(-1., 1., size=[m, n])


def generator(z):
  G_H1 = tf.nn.relu(tf.matmul(z, G_W1) + G_B1)
  G_log_prob = tf.matmul(G_H1, G_W2) + G_B2
  G_prob = tf.nn.sigmoid(G_log_prob)

  return G_prob

def discriminator(x):
  D_H1 = tf.nn.relu(tf.matmul(x, D_W1) + D_B1)
  D_logit = tf.matmul(D_H1, D_W2) + D_B2
  D_prob = tf.nn.sigmoid(D_logit)

  return D_prob, D_logit

def plot(samples):
  fig = plt.figure(figsize=(4, 4))
  gs = gridspec.GridSpec(4, 4)
  gs.update(wspace=0.05, hspace=0.05)

  for i, sample in enumerate(samples):
    ax = plt.subplot(gs[i])
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

  return fig



def main(_):
  # Import data
  input_digit = -1
  while input_digit > 9 or input_digit < 0:
    input_digit = int(input("number that you want me to learn:"))

  print("I am going to learn {}".format(input_digit))
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  ##eg digit '1'
  given_Digit = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
  given_Digit[input_digit] = 1.0
  ##get the list of images whose labels equals to given_Digit
  train_images = []
  train_labels = []
  length = len(mnist.train.images)
  for index in range(length):
    image = mnist.train.images[index]
    label = mnist.train.labels[index]
    if  np.array_equal(label, given_Digit):
      train_images.append(image)
      train_labels.append(given_Digit)

  # Create the model
  G_sample = generator(Z)
  D_real, D_logit_real = discriminator(X)
  D_fake, D_logit_fake = discriminator(G_sample)

  D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1.0 - D_fake))
  G_loss = -tf.reduce_mean(tf.log(D_fake))

  D_solver = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(D_loss, var_list=theta_D)
  G_solver = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(G_loss, var_list=theta_G)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  batch_size = 128
  start = 0
  end = batch_size
  Z_dim = 100;
  train_image_size = len(train_images)
  i=0
  # Train
  for step in range(10000):
    if step > train_image_size :
      break
    batch_xs = np.asarray(train_images[start:end])
    if step % batch_size == 0:
      start += batch_size
      if(end+batch_size > train_image_size):
        end = train_image_size - 1
      else:
        end += batch_size


      samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})
      fig = plot(samples)
      plt.savefig('out/{}.png'.format(str(i).zfill(3)),bbox_inches='tight')
      i += 1
      plt.close(fig)

    _, cur_D_loss = sess.run([D_solver, D_loss], feed_dict={X : batch_xs, Z:sample_Z(batch_size, Z_dim)})
    _, cur_G_loss = sess.run([G_solver, G_loss], feed_dict={Z:sample_Z(batch_size, Z_dim)})
    if step % batch_size == 0:
      print('Iter: {}'.format(step))
      print('start: {}'.format(start))
      print('end: {}'.format(end))
      print('D loss: {:.4}'. format(cur_D_loss))
      print('G_loss: {:.4}'.format(cur_G_loss))
      print()
















if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)