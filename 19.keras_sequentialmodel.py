import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# create data
X = np.linspace(-2,2,200)
np.random.shuffle(X) # randomize the data
# add some noise data
Y = 0.5*X+2+np.random.normal(0,0.05,(200,))

# show the input data
plt.scatter(X,Y)
plt.show()

# divide data into training and testing data
X_train,Y_train = X[:160],Y[:160]
X_test,Y_test = X[160:],Y[160:]

# construct the model
model = Sequential()
model.add(Dense(units=1,activation='relu',input_dim=1))

# compile the model
model.compile(loss='mse',optimizer='sgd')

# train the model
model.fit(X_train,Y_train,epochs=100,verbose=0,batch_size=64)

# test the model
print('\nTesting---------')
cost = model.evaluate(X_test,Y_test,batch_size=40)
print('test cost:',cost)
W,b = model.layers[0].get_weights()
print('Weights=',W,'\nbiases=',b)

# visulization
# plotting the prediction
Y_pred = model.predict(X_test)
plt.scatter(X_test,Y_test)
plt.plot(X_test,Y_pred)
plt.show()