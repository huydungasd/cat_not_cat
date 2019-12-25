import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from typing import Tuple
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset


# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Example of a picture
# index = 25
# plt.imshow(train_set_x_orig[index])
# plt.show()
# print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")


#==================================================================================
# Data processing
# Visualization of data's properties
print("training shape : " + str(train_set_x_orig.shape))
print("test shape : " + str(test_set_x_orig.shape))
print("training samples : " + str(train_set_x_orig.shape[0]))
print("test samples : " + str(test_set_x_orig.shape[0]))
print("dimension of each picture : " + str(train_set_x_orig.shape[1:]) + "\n\n")

# Reshape data
train_set_x_flatten = train_set_x_orig.reshape((train_set_x_orig.shape[0], -1)).T
test_set_x_flatten = test_set_x_orig.reshape((test_set_x_orig.shape[0], -1)).T

print("training flatten shape : " + str(train_set_x_flatten.shape))
print("test flatten shape : " + str(test_set_x_flatten.shape) + "\n\n")

# Standardize data
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

#==================================================================================
# Define activation function
def sigmoid(z : np.array) -> np.array:
    return 1/(1 + np.exp(-z))

def d_sigmoid(z : np.array) -> np.array:
    sig = sigmoid(z)
    return sig * (1 - sig)

def cost(yhat : np.array, y : np.array) -> float:
    assert(yhat.shape[0] == 1)
    assert(y.shape[0] == 1)
    assert(yhat.shape[1] == y.shape[1])
    return -np.mean(y*np.log(yhat) + (1 - y)*np.log(1 - yhat), axis=1)

# Initialize parameters
def initialization(dim : int) -> Tuple:
    w = np.zeros((dim, 1))
    b = 0
    return w, b

# Propagation
def propagate(  w : np.array,
                b : float,
                X_train : np.array,
                Y_train : np.array) -> Tuple:

    # Number of samples
    m = X_train.shape[1]

    # Forward probagation
    yhat = sigmoid(w.T @ X_train + b)
    c = cost(yhat, Y_train)

    # Back probagation
    dw = 1/m * X_train @ (yhat - Y_train).T
    db = np.mean(yhat - Y_train)

    grads = {   'dw': dw,
                'db': db}

    return c, grads

# Optimisation
def optimize(   X_train,
                Y_train,
                learning_rate,
                num_iterations):
    
    w, b = initialization(X_train.shape[0])
    for i in range(num_iterations):
        cost, grads = propagate(w, b, X_train, Y_train)
        dw = grads['dw']
        db = grads['db']
        w = w - learning_rate*dw
        b = b - learning_rate*db
        if i % 100 == 0:
            print('Cost after {0} iterations : {1}'.format(i, cost))
    
    params = {  'w' : w,
                'b' : b}
    return params

# Prediction
def predict(w, b, X):
    yhat = sigmoid(w.T @ X + b)
    Y_predict = np.zeros((1, yhat.shape[1]))
    Y_predict[0, yhat[0, :] > 0.5] = 1
    Y_predict[0, yhat[0, :] <= 0.5] = 0

    return Y_predict

# model
def model(X_train, Y_train, X_test, Y_test, learning_rate=0.005, num_iterations=2000):
    params = optimize(X_train, Y_train, learning_rate, num_iterations)
    w = params['w']
    b = params['b']

    Y_train_predict = predict(w, b, X_train)
    Y_test_predict = predict(w, b, X_test)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_train_predict - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_test_predict - Y_test)) * 100))

    return 0

model(train_set_x, train_set_y, test_set_x, test_set_y)
