import tensorflow as tf

# create weights and bias of model
weights = tf.Variable(tf.random_normal([784,200],stddev=0.35),name='weights')
bias = tf.Variable(tf.zeros([200]),name='bias')

# initialize weights and bias
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

# save variables of the model
saver = tf.train.Saver()
saver.save(sess, '.tmp/model/',global_step=100)

# recover models variables
saver = tf.train.import_meta_graph('.tmp/model/-100.meta')
saver.restore(sess, tf.train.latest_checkpoint('./tmp/model/'))

# print restored variables
print(sess.run('biases:0'))