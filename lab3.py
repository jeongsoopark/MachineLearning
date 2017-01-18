import tensorflow as tf
import numpy as np


xy = np.loadtxt('lab3', unpack=true, dtype='float32')

x_data = xy[0:-1]
y_data = xy[-1]

W= tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

