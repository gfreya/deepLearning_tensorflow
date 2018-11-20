import tensorflow as tf

labels = [0,2]
logits = [[2,0.5,1],
          [0.1,1,3]]

logits_scaled = tf.nn.softmax(logits)
result1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits)

with tf.Session() as sess:
    print(sess.run(result1))