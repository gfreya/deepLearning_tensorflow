import numpy as np
import tensorflow as tf

input_data = tf.Variable(np.random.rand(3,3),dtype = tf.float32)
# np.random,rand() shape, return array between[0,1]
output = tf.nn.weighted_cross_entropy_with_logits(logits=input_data,targets=[[1.0,0.0,0.0],[0.0,0.0,1.0],[0.0,0.0,1.0]],pos_weignt=2.0)

with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(output))