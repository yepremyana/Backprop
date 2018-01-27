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

#minibatch of 128
def minibatch(train_set, train_labs, n, batch_size):
    sample = train_set[n*batch_size:(n*batch_size + batch_size)]
    label = train_labs[n*batch_size:(n*batch_size + batch_size)]
    return sample,label

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
    a = np.array(x_array)
    x_bias = np.insert(a, 0, 1, axis = 1)
    return [np.append(x,1) for x in x_array]

def one_hot_encoding(label_to_1hotencode):
    encoded_list = list()
    for label in label_to_1hotencode:
        label_zero = [0 for _ in xrange(10)]
        label_zero[label] = 1
        encoded_list.append(label_zero)
    return encoded_list

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
    w_hidden_output = w_hidden_output[1:]
    w_input_hidden = w_input_hidden[1:]

    #w_ij
    activation_ih = activation(input_data, w_input_hidden)
    error = sigmoid(activation_ih, derivative = True)
    c = error * np.dot(delta_k, w_hidden_output.T)
    w_input_hidden += np.dot(input_data.T, c)

    #w_jk
    z = softmax(output_ih)
    w_hidden_output += np.dot(z.T, delta_k)

    return w_hidden_output, w_input_hidden

def hold_out(train_im, train_lab, percent):
    num_hold_out = int(np.round(1/float(percent) * len(train_im)))
    hold_out_im = train_im[-num_hold_out:]
    hold_out_labels = train_lab[-num_hold_out:]
    train_im = train_im[:-num_hold_out]
    train_lab = train_lab[:-num_hold_out]
    return hold_out_im, hold_out_labels, train_im, train_lab


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
input_mini, label_mini = minibatch(train_dat, train_lab, 0, 128)
cat_mini = one_hot_encoding(label_mini)
input_bias = add_bias_term(input_mini)

final_ho, final_ih = forward(w_ih, w_ho)
w_ho, w_ih = backprop(input_mini,cat_mini,final_ho, final_ih, w_ho, w_ih)

#1. add learning rate
#2. make epochs
#3. do gradient checker
#4. make into a class
#5. fix the weights so that they are 64
#6. make w_ij and w_jk functions
