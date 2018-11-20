import tensorflow as tf

# output of neural network
logits = tf.constant([1.0,2.0,3.0],[1.0,2.0,3.0],[1.0,2.0,3.0])
# softmax layer
y = tf.nn.softmax(logits)
# correct label: only one 1
y_ = tf.constant([0.0,0.0,1.0],[1.0,0.0,0.0],[1.0,0.0,0.0])
# calculate crossentropy
cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
# use tf.nn.softmax_cross_entropy_with_logits() directly calculate the entropy
# remember use the tf.reduce_sum
cross_entropy2 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=logits))

with tf.Session() as sess:
    softmax = sess.run(y)
    ce = sess.run(cross_entropy)
    ce2 = sess.run(cross_entropy2)
    print("softmax result=", softmax)
    print("cross_entropy result=",ce)
    print("softmax_cross_entropy_with_logits result=",ce2)