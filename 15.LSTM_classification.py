from tensorflow.examples.tutorials.mnist import input_data

data_dir = './MNIST_data/'
mnist = input_data.read_data_sets(data_dir,one_hot=True)
print(mnist.train.images.shape)
print(mnist.train.labels.shape)
print(mnist.test.images.shape)
print(mnist.test.labels.shape)

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

plt.show(mnist.train.images[0].reshape((28,28)),cmap='gray')
plt.title('%i'%np.argmax(mnist.train.labels[0]))
plt.show()

# construction of model
TIME_STEP = 28 # read 28 rows of an image
INPUT_SIZE = 28 # the len of vector(28 xiangsu each row)

# define input,output
tf_x = tf.placeholder(tf.float32,[None,TIME_STEP*INPUT_SIZE])
image = tf.reshape(tf_x,[-1,TIME_STEP,INPUT_SIZE])
tf_y = tf.placeholder(tf.int32,[None,10])

# define structure of LSTM
rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=64)
# construct network
outputs,(h_c,h_n) = tf.nn.dynamic_rnn(
    rnn_cell,
    image,
    initial_state = None,
    dtype = tf.float32,
    time_major = False,
)

# final output: the output of the last time
output = tf.layers.dense(outputs[:,-1,:],10)

LR = 0.01
loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y,logits=output)

train_op = tf.train.AdamOptimizer(LR).minimize(loss)

