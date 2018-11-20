# load and process data
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.INFO)
# read data
mnist = input_data.read_data_sets('./MNIST_data/',one_hot=False)

#define input feature col
feature_columns = [tf.feature_column.numeric_column("image",shape=[784])]

# create two hidden layers, nodes:200,50
estimator = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                       hidden_units=[200, 50],
                                       optimizer = tf.train.AdamOptimizer(1e-4),
                                       n_classes = 10,
                                       dropout=0.2,
                                       model_dir="model_dir")

# train model
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"image":mnist.train.images},
    y=mnist.train.labels.astype(np.int32),
    num_epochs=None,
    batch_size=128,
    shuffle=True
)
estimator.train(input_fn=train_input_fn,steps=20000)

# test the model
test_input_fn= tf.estimator.inputs.numpy_input_fn(
    x={"image":mnist.train.images},
    y=mnist.train.labels.astype(np.int32),
    num_epochs=1,
    batch_size=128,
    shuffle=False
)
test_results = estimator.evaluate(input_fn=test_input_fn)
accuracy_score = test_results["accuracy"]
print("\nTest Accuracy: {0:.4f}\n".format(accuracy_score))
print(test_results)
