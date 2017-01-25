import tensorflow as tf


#input date set
x_data = [1., 2., 3., 4.]
#y_data = [1., 2., 3., 4.]
y_data = [1., 3., 5., 7.]

W= tf.Variable(tf.random_uniform([1], -100.0, 1000.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# hypothesis
hypothesis = W * X

cost = tf.reduce_mean(tf.square(hypothesis - Y))

#minimize
descent = W - tf.mul(0.1, tf.reduce_mean(tf.mul((tf.mul(W,X) - Y) , X)))
update = W.assign(descent)


#initialize before running
init = tf.global_variables_initializer()

#session launch
sess = tf.Session()
sess.run(init)

#fit

for step in range(20):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print(step, sess.run(cost, feed_dict={X:x_data, Y: y_data}), sess.run(W))

print(sess.run(hypothesis, feed_dict={X:5}))
print(sess.run(hypothesis, feed_dict={X:2.5}))
