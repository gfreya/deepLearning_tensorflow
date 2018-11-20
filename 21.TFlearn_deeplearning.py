from __future__ import division,print_function,absolute_import
import tflearn
import tensorflow as tf
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist

X,Y,testX,testY = mnist.load_data(one_hot=True)
X = X.reshape([-1,28,28,1])
testX = testX.reshape([-1,28,28,1])

tf.reset_default_graph()
network = input_data(shape=[None,28,28,1],name='input')
# define conv
network = conv_2d(network,32,3,activation='relu',regularizer="L2")
# define max_pooling
network = max_pool_2d(network,2)
# normalization
network = local_response_normalization(network)
network = conv_2d(network,64,3,activation='relu',regularizer="L2")
network = max_pool_2d(network,2)
network = local_response_normalization(network)
# full connected
network = fully_connected(network,128,activation='tanh')
# dropout
network = dropout(network,0.8)
network = fully_connected(network,256,activation='tanh')
network = dropout(network,0.8)
network = fully_connected(network,10,activation='tanh')
# regression
network = regression(network,optimizer='adam',learning_rate=0.01,loss='categorical_crossentropy',name='target')
# DNN
model = tflearn.DNN(network,tensorboard_verbose=0)
model.fit({'input':X},{'target':Y},n_epoch=20,
          validation_set=({'input':testX},{'target':testY}),
          snapshot_step=100,show_metric=True,run_id='convnet_mnist')