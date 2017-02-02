import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')

x_data = xy[0:-1]
y_data = xy[-1]

W= tf.Variable(tf.random_uniform([1,3], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# hypothesis
hypothesis = tf.matmul(W, x_data)

#simple cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

#minimize
a = tf.Variable(0.1) # alpha, learning rate
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)



#fit the line

for step in range(2001):
    #sess.run(train)
    sess.run(train)
    if(step % 20) == 0:
        print(step, sess.run(cost), sess.run(W) )
