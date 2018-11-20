import os
import struct
import numpy as np
import tensorflow as tf

def dense_to_one_hot(labels_dense,num_classes=10):
    """class labels-->one hot vector"""
    num_labels = labels_dense.shape[0]
    index_offset = np.array(num_labels)*num_classes
    labels_one_hot = np.zeros((num_labels,num_classes))
    labels_one_hot.flat[index_offset+labels_dense.ravel()]=1
    return labels_one_hot

# load function
def load_mnist(path,kind='train'):
    """load dataset cording to assigned path"""
    labels_path = os.path.join(path,'%s-labels-idx1-ubyte'%kind)
    images_path = os.path.join(path,'%s-labels-idx3-ubyte'%kind)

    with open(labels_path,'rb') as lbpath:
        magic,n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath,dtype=np.uint8)
        labels = dense_to_one_hot(labels)

    with open(images_path,'rb') as imgpath:
        magic,num,rows,cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels),784)

    return images, labels

# define parameters of the network
# parameters
learning_rate = 0.001
num_steps = 4000
batch_size = 128
display_step = 100
# network parameters
n_input = 784 # mnist dataset dim: 28x28
n_classes = 10 # classification of mnist
dropout = 0.80 # probability of saving nodes

# get the data
X_train, y_train = load_mnist('./MNIST_data/',kind='train')
print('Rows:%d,columns:%d'%(X_train.shape[0],X_train.shape[1]))
print('Rows:%d,columns:%d'%(y_train.shape[0],y_train.shape[1]))

X_test,y_test = load_mnist('./MNIST_data/',kind='t10k')
print('Rows:%d,columns:%d'%(X_test.shape[0],X_test.shape[1]))

#create Dataset
sess = tf.Session()
# create data tensor from images and labels
dataset = tf.data.Dataset.from_tensor_slices(
    (X_train.astype(np.float32),y_train.astype(np.float32))
)
# batch
dataset = dataset.batch(batch_size)
# create iterator
iterator = dataset.make_initializable_iterator()
# two placeholders
_data = tf.placeholder(tf.float32,[None,n_input])
_labels = tf.placeholder(tf.float32,[None,n_classes])
# initialize
sess.run(iterator.initializer,feed_dict={_data:X_train.astype(np.float32),
                                         _labels:y_train.astype(np.float32)})
# get the input
X, Y = iterator.get_next()

# create the model
def conv_net(x,n_classes,dropout,reuse,is_training):
    # define scope of reused parameters
    with tf.variable_scope('ConvNet',reuse=reuse):
        # reshape: input dim 1 --> dim 4
        # dim 4 [Batch Size, Height, Width, Channel]
        x = tf.reshape(x,shape=[-1,28,28,1])

        # define conv1 layer, 16 filters 5x5
        conv1 = tf.layers.conv2d(x,16,5,activation=tf.nn.relu)
        # max pooling: strides:2 2x2
        conv1 = tf.layers.max_pooling2d(conv1,2,2)

        # define conv2 layer, 36 filters 3x3
        conv2 = tf.layers.conv2d(conv1,36,3,activation=tf.nn.relu)
        # define max pooling: strides:2 kernel:2x2
        conv2 = tf.layers.max_pooling2d(conv2,2,2)

        # flattern the data to a 1_D vector for the full connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # define full_connected layer
        fc1 = tf.layers.dense(fc1,128)
        # define dropout
        fc1 = tf.layers.dropout(fc1,rate=dropout,training=is_training)

        # output layer
        out = tf.layers.dense(fc1,n_classes)
        # use softmax to classify
        out = tf.nn.softmax(out) if not is_training else out

    return out

# train and evaluate the model
# create calculation graph
logits_train = conv_net(X,n_classes,dropout,reuse=False,is_training=True)
logits_test = conv_net(X, n_classes, dropout, reuse=True, is_training=False)

# define loss function
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits_train,labels=Y
))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# evaluate model, but not execute dropout
correct_pred = tf.equal(tf.argmax(logits_test,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

# initialize parameters
init = tf.global_variables_initializer()
sess.run(init)
for step in range(1,num_steps+1):
    try:
        # train model
        sess.run(train_op)
    except tf.errors.OutOfRangeError:
        #when reading the end of the dataset, reload iterator
        sess.run(iterator.initializer,
                 feed_dict={_data:X_train.astype(np.float32),
                            _labels:y_train.astype(np.float32)})
        sess.run(train_op)

    if step%display_step == 0 or step == 1:
        # calculate loss function and precision
        # data is different
        loss, acc = sess.run([loss_op,accuracy])
        print("Step"+str(step)+",Minibatch Loss= "+ \
              "{:.4f}".format(loss)+",Training Accuracy="+\
              "{:.3f}".format(acc))
