import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

# define hyper-parameter
BATCH_SIZE = 64
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.01

# read data
mnist = input_data.read_data_sets('./MNIST_data/',one_hot=True)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]

# to understande better, get one image
print(mnist.train.images.shape)  #(55000,28*28)
print(mnist.train.labels.shape)  #(55000,10)
plt.show(mnist.train.images[0].reshape((28,28)),cmap='gray')
plt.title('%i'%np.argmax(mnist.train.labels[0]))
plt.show()

# define vector x(placeholder)
tf_x = tf.placeholder(tf.float32,[None,TIME_STEP*INPUT_SIZE])
image = tf.reshape(tf_x,[-1,TIME_STEP,INPUT_SIZE])
#define vector y(placeholder)
tf_y = tf.placeholder(tf.int32,[None,10])
# RNN recurrent structure, using LSTM
rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=64)
outputs,(h_c,h_n) = tf.nn.dynamic_rnn(
    rnn_cell,
    image,
    initial_state=None,
    dtype=tf.float32,
    time_major=False,
)
# using last output as final output
output = tf.layers.dense(outputs[:,-1,:],10)
# calculate loss function
loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

# calculate the precision
accuracy = tf.metrics.accuracy(labels=tf.argmax(tf_y,axis=1),predictions=tf.argmax(output,axis=1),)[1]

sess = tf.Session()

# initialization
init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
sess.run(init_op)

# training
for step in range(2000):
    b_x,b_y = mnist.train.next_batch(BATCH_SIZE)
    _,loss_ = sess.run([train_op,loss],{tf_x:test_x,tf_y:test_y})
    if step%50==0:
        accuracy_ = sess.run(accuracy,{tf_x:test_x,tf_y:test_y})
        print('train loss: %.4f'%loss_,'|test accuracy:%.2f'%accuracy)

# output 10 predictions
test_output = sess.run(output,{tf_x:test_x[:10]})
pred_y = np.argmax(test_output,1)
print(pred_y,'prediction number')
print(np.argmax(test_y[:10],1),'real number')