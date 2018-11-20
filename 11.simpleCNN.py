import tensorflow as tf

# 3x3 image: shape [1,3,3,1]
# filter(kernel): shape [1,1,1,1]
# strides = 1
# feature map 3x3
input = tf.Variable(tf.random_normal([1,3,3,5]))
filter = tf.Variable(tf.random_normal([1,1,5,1]))
conv2d_1 = tf.nn.conv2d(input,filter,strides=[1,1,1,1],padding='VALID')

# 5x5 image
# filter 3x3
# strides 1
# feature map 3x3
input = tf.Variable(tf.random_normal([1,5,5,5]))
filter = tf.Variable(tf.random_normal([3,3,5,1]))
conv2d_2 = tf.nn.conv2d(input,filter,strides=[1,1,1,1],padding='VALID')
# if padding='SAME', feature map 5x5
conv2d_3 = tf.nn.conv2d(input,filter,strides=[1,1,1,1],padding='SAME')

# if strides!=1, dim:2
# strides = [1, image_height_stride, image_width_stride,1]
# out_channel = 3
# output: feature map 3x3
input = tf.Variable(tf.random_normal([1,5,5,5]))
filter = tf.Variable(tf.random_normal([3,3,5,3]))
conv2d_4 = tf.nn.conv2d(input,filter,strides=[1,2,2,1],padding='SAME')
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print("Example1:")
    print(sess.run(conv2d_1))
    print("Example2:")
    print(sess.run(conv2d_2))
    print("Example3:")
    print(sess.run(conv2d_3))
    print("Example4:")
    print(sess.run(conv2d_4))


