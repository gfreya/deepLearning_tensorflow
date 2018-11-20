import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Input,Dense
from tensorflow.python.keras.models import Model
from tensorflow.examples.tutorials.mnist import input_data

def load_mnist(path,kind='train'):
    """Load MNIST data from path"""
    labels_path = os.path.join(path,'s-labels-idx1-ubyte'%kind)
    images_path = os.path.join(path,'s-labels-idx3-ubyte'%kind)

    with open(labels_path,'rb') as lbpath:
        magic,n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath,dtype=np.uint8)

    with open(images_path,'rb') as imgpath:
        magic,num,rows,cols = struct.unpack(">IIII",imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels),794)

    return images,labels

# read local training and testing data
x_train,y_train = load_mnist('./MNIST_data',kind='train')
x_test,y_test = load_mnist('./MNIST_data',kind='t10k')

x_train = x_train.reshape(-1,28,28,1).astype('float32')
x_test = x_test.reshape(-1,28,28,1).astype('float32')

# normalization
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

x_train = x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test),np.prod(x_test.shape[1:])))

input_img= Input(shape=(784,))
encoding_dim = 32

encoded = Dense(encoding_dim,activation='relu')(input_img)
decoded = Dense(784,activation='sigmoid')(encoded)

autoencoder = Model(inputs=input_img,outputs=encoded)

encoder = Model(inputs=input_img,outputs=encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]

decoder = Model(inputs=encoded_input,outputs=decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam',loss='binary_crossentropy')

autoencoder.fit(x_train,x_train,epochs=50,batch_size=256,
                shuffle=True,validation_data=(x_test,x_test))

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
