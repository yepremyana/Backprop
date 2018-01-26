from mnist import MNIST
import numpy as np
import math
import random
import pdb
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Will load MNIST data and return the first and last specified number of training & testing images respectively
def load_data(num_train, num_test, directory='mnist'):
    mndata = MNIST(directory)
    train_dat, train_lab = mndata.load_training()
    test_dat, test_lab = mndata.load_testing()
    return np.array(train_dat[:num_train]), np.array(train_lab[:num_train]), \
           np.array(test_dat[-num_test:]), np.array(test_lab[-num_test:])

# Images /127.5 - 1 so that they are in range [-1,1]
def z_score_data(train_dat, test_dat):
    train_dat = train_dat/127.5 -1
    test_dat = test_dat/127.5 - 1
    return train_dat, test_dat

##<editor-fold desc="Forward Propagation">
##</editor-fold>

#minibatch of 128
def minibatch(f, n):
    return random.sample(f, n)

def activation(x, w):
    return np.dot(x, w)

def sigmoid(h, derivative = False):
    if (derivative == True):
        y = 1 / (1 + np.exp(-1 * h))
        return y*(1-y)
    else:
        return 1 / (1 + np.exp(-1 * h))

#Add a 1 in front of every input vector that accounts for the bias weight
def add_bias_term(x_array):
    return [np.append(x,1) for x in x_array]

def softmax(j):
    ak = np.exp(j)
    sum_ak = np.sum(ak, 1)
    sum_ak = np.reshape(sum_ak, (ak.shape[0], 1))
    sum_ak = np.repeat(sum_ak, ak.shape[1], axis=1)
    return ak / (1.0 * sum_ak)

#feedforward
def forward(w_input_hidden, w_hidden_output):

    #input to hidden
    input_ih = activation(input_bias, w_input_hidden)
    output_ih = sigmoid(input_ih)
    output_ih_bias = add_bias_term(output_ih)

    #hidden to output
    input_ho = activation(output_ih_bias, w_hidden_output)
    output_ho = softmax(input_ho)

    return output_ho, output_ih

#backpropogation
def backprop(input_data, t, output_ho,output_ih, w_hidden_output, w_input_hidden):

    #where t is the expected and y is the output (from forwards_prop)
    delta_k = t - output_ho

    #w_jk
    z = softmax(output_ih)
    w_hidden_output += np.dot(z.T, delta_k)

    #w_ij
    error = delta_k.T * sigmoid(output_ih, derivative = True)
    c = np.dot(error, w_hidden_output.T)
    w_input_hidden += np.dot(input_data.T, c)

    return w_hidden_output, w_input_hidden

# IMPLEMENTATION:

# 1. Load Data
num_train = 10000
num_test = 1000
tr_i, tr_l, test_i, test_l = load_data(num_train, num_test)

# 2. Z-score data
tr_i, test_i = z_score_data(tr_i, test_i)

#initializing weights
mu, sigma = 0, 0.1
w_ih = np.random.normal(mu, sigma, (785,64))
w_ho = np.random.normal(mu, sigma, (65,10))

#create minibatch
#i think i messed this up
input_mini = minibatch(train_dat,128)
#fix this
input_bias = add_bias_term(input_mini)

final_ho, final_ih = forward(w_ih, w_ho)
#w_ho, w_ih = backprop( , ,final_ho, final_ih, w_ho, w_ih)

#1. add learning rate
#2. add holdout
#3. do gradient checker
#4. make into a class
#5. check mini batches -- probably add tuples to include the labels
