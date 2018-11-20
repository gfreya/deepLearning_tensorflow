import tensorflow as tf

sess = tf.InteractiveSession()

# create a 4x4 matrics
matrix_input = tf.Variable(tf.random_normal([4,4],mean=0.0,stddev=1.0))

# initialize variables
tf.global_variables_initializer().run()

# print original matrix
print("original matrix:\n", matrix_input.eval())

# using relu to activate original matrix
matrix_output = tf.nn.relu(matrix_input)
# print activated matrix
print("activated matrix:\n", matrix_output.eval())
