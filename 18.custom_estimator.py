import argparse
import os
import numpy as np
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SZIE = 100
DATA_DIR = './MNIST_data/'
MODEL_DIR = os.path.join("./custom_model_dir",str(int(time.time())))

NUM_STEPS = 1000
tf.logging.set_verbosity(tf.logging.INFO)
print("using model dir: %s" %MODEL_DIR)

# define model
def cnn_model_fn(features,labels,mode):

    # input layer
    # reshape X to 4-D tensor:[batch_size,width,height,channels]
    # MNIST:28x28,one channel
    input_layer = tf.reshape(features["x"],[-1,28,28,1])

    # conv1
    # 32 5x5 kernels, activation: ReLU
    # padding: SAME
    # input tensor:[batch_size,28,28,1], output tensor:[batch_size,28,28,32]
    conv1 = tf.layers.conv2d(
        inputs = input_layer,
        filters = 32,
        kernel_size=[5,5],
        padding = "same",
        activation = tf.nn.relu
    )
    conv1 = tf.layers.batch_normalization(inputs=conv1,training=mode==tf.estimator.ModeKeys.TRAIN,name='BN1')

    # pooling
    # 2x2, strides=2
    # input:[batch_size,14,14,32]
    # output: [batch_size,14,14,32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2],strides=2)

    # conv2
    # 64 5x5
    # padding: same
    # input[batch_size,14,14,32]
    # output[batch_size,14,14,64]
    conv2 = tf.layers.conv2d(
        inputs = pool1,
        filters = 64,
        kernel_size=[5,5],
        padding="same",
        activation = tf.nn.relu
    )
    # pooling2
    # 2x2,strides=2
    # input[batch_size,14,14,64]
    # output[batch_size,7,7,64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=2)

    # tensor flatten to vector
    # input[batch_size,7,7,64]
    # output[batch_size,7*7*64]
    pool2_flat = tf.reshape(pool2,[-1,7*7*64])

    # full-connected
    # 1024 nurons
    # input[batch_size,7,7,64]
    # output[batch_size,1024]
    dense = tf.layers.dense(inputs=pool2_flat,units=1024,activation=tf.nn.relu,name='dense1')

    # add a dropout,0.6
    dropout = tf.layers.dropout(
        inputs=dense,rate=0.4,training=mode==tf.estimator.ModeKeys.TRAIN
    )

    # logits layer
    # input[batch_size,1024]
    # output[batch_size,10]
    logits = tf.layers.dense(inputs=dropout,units=10)

    predictions={
        # predicted value
        "classes":tf.argmax(input=logits,axis=1),
        # 'softmax_tensor' add to data flow
        "probabilities":tf.nn.softmax(logits,name='softmax_tensor')
    }
    prediction_output = tf.estimator.export.PredictOutput({"classes":tf.argmax(input=logits,axis=1),"probabilities":tf.nn.softmax(logits,name="softmax_tensor")})
    if mode==tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions,export_outputs={tf.save_model.signiture_constants.DEFAULT_SERVING_SIGNITURE_DEF_KEY:prediction_output})

    # calculate loss function
    onehot_labels = tf.one_hot(indices=tf.cast(labels,tf.int32),depth=10)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,logits=logits)
    # generate one summary info
    tf.summary.scalar('loss',loss)
    tf.summary.histogram('conv1',conv1)
    tf.summary.histogram('dense',dense)

    # Configure thr Training Op(for Train Mode)
    if mode==tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)

    # add evaluation
    eval_metric_op = {
        "accuracy":tf.metrics.accuracy(
            labels=labels,predictions=predictions["classes"]
        )
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,loss=loss,eval_metric_ops=eval_metric_op
    )

# define input func
def generate_input_fn(dataset,batch_size=BATCH_SZIE):
    def _input_fn():
        X = tf.constant(dataset.images)
        Y = tf.constant(dataset.labels,dtype=tf.int32)
        image_batch,label_batch=tf.train.shuffle_batch([X,Y],
            batch_size=batch_size,
            capacity=8*batch_size,
            min_after_dequeue=4*batch_size,
            enqueue_many=True
                                                       )
        return{'x':image_batch},label_batch
    return _input_fn

# load data
mnist = input_data.read_data_sets(DATA_DIR)
train_data = mnist.train.images # Return np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

eval_data = mnist.test.images # return np.array
eval_labels = np.asarray(mnist.test.labels,dtype=np.int32)

predict_data_batch = mnist.test.next_batch(10)

# create estimator
mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn,model_dir=MODEL_DIR)

tensor_to_log = {"probabilities":"softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
    tensors=tensor_to_log,every_n_iter=2000
)

# train the model
mnist_classifier.train(
    input_fn=generate_input_fn(mnist.train,batch_size=BATCH_SZIE),
    steps = NUM_STEPS,
    hooks=[logging_hook]
)

# test the model
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x":eval_data},
    y = eval_labels,
    num_epochs=1,
    shuffle=False
)
eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)

# save model
def serving_input_receiver_fn():
    feature_tensor = tf.placeholder(tf.float32,[None,78])
    return tf.estimator.export.ServingInputReceiver({'x':feature_tensor},{'x':feature_tensor})

exported_model_dir = mnist_classifier.export_saved_model(MODEL_DIR,serving_input_receiver_fn)
decoded_model_dir = exported_model_dir.decode("utf-8 ")
