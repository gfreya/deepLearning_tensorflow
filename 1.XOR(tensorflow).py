import tensorflow as tf
import numpy as np

# define input and objective values
X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0],[1],[1],[0]])

# define placeholder, get the data directly from X and Y
x = tf.placeholder(tf.float32,[None,2])
y = tf.placeholder(tf.float32,[None,1])

# initialize the weight, norm-distribution
# w1->input to hidden, w2->hidden to output
w1 = tf.Variable(tf.random_normal([2,2]))
w2 = tf.Variable(tf.random_normal([2,1]))

# define bias, b1->h, b2->O
b1 = tf.Variable([0.1,0.1])
b2 = tf.Variable(0.1)

# activation
h = tf.nn.relu(tf.matmul(x,w1)+b1)
# output value
out = tf.matmul(h,w2)+b2

# define loss function
loss = tf.reduce_mean(tf.square(out-y))
# adam
train = tf.train.AdamOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        sess.run(train,feed_dict={x:X,y:Y})
        loss_ = sess.run(loss,feed_dict={x:X,y:Y})
        if i%200==0:
            print("step:%d,loss:%.3f"%(i,loss_))

    print("X: %r"%X)
    print("pred: %r"%sess.run(out,feed_dict={x:X}))