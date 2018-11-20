import tensorflow as tf

# create a new data-stream graph
graph = tf.Graph()

with graph.as_default():
    a = tf.add(2,4)
    b = tf.multiply(2,4)

# more than one graph
graph1 = tf.Graph()
graph2 = tf.Graph()

with graph1.as_default():
    #define op, tensor
    pass

with graph2.as_default():
    pass