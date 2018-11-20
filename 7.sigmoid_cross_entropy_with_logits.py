import tensorflow as tf
import numpy as np

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

# define a tri-classification problem with 5 samples
y = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1]])
logits = np.array([[10,3.8,2],[8,10.9,1],[1,2,7.4],[4.8,6.5,1.2],[3.3,8.8,1.9]])

# calculate the results with sigomoid()
y_pred = sigmoid(logits)
output1 = -y*np.log(y_pred)-(1-y)*np.log(1-y_pred)
print('self_define_output1:',output1)

with tf.Session() as sess:
    y = np.array(y).astype(np.float64)
    output2 = sess.run(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=logits))
    print('sigmoid_cross_entropy_with_logits:', output2)

    reduce_mean = sess.run(tf.reduce_mean(output2))
    print('reduce_mean:',reduce_mean)