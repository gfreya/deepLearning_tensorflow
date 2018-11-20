import tensorflow as tf

# construct calculation graph
a = tf.constant(1.,name="a")
b = tf.constant(3.,shape=[2,2],name="b")

# create a session
sess = tf.Session()

# execute the session
result_a = sess.run([a,b])
print("result_a:", result_a[0])
print("result_b:", result_a[1])