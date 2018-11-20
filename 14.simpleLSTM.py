import tensorflow as tf

num_units = 128
num_layers = 2
batch_size = 100

# create a BasicLSTMCell, LSTM recurrent
# num_units: number of cells in BasicLSTMCell, number more, features more
rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units)
# using multi-layer structure, return also cell structure
if num_layers>=2:
    run_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell]*num_layers)

# define initial status
initial_state = rnn_cell.zero_state(batch_size,dtype=tf.float32)
input_data = [] # not defined

# define input data's structure
outputs, state = tf.nn.dynamic_rnn(rnn_cell,input_data,initial_state=initial_state,dtype=tf.float32)