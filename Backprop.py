from mnist import MNIST
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#Will load MNIST data and return the first and last specified number of training & testing images respectively
mndata = MNIST('mnist')
train_dat, train_lab = mndata.load_training()
test_dat, test_lab = mndata.load_testing()

#divide by 127.5 so that they are in range [0...2]
train_dat = np.array(train_dat, dtype=np.float32)
train_dat /= 127.5
train_dat = train_dat - 1
test_dat = np.array(test_dat, dtype=np.float32)
test_dat /= 127.5
test_dat = test_dat - 1

#initializing weights
mu, sigma = 0, 0.1
w_ih = np.random.normal(mu, sigma, (785,64))
w_ho = np.random.normal(mu, sigma, (65,10))

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
def forward(input_bias, w_input_hidden, w_hidden_output):

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
    gradient_ih = np.dot(input_data.T, c)
    w_input_hidden += gradient_ih

    #w_jk
    z = softmax(output_ih)
    gradient_ho = np.dot(z.T, delta_k)
    w_hidden_output += gradient_ho

    return w_hidden_output, w_input_hidden, gradient_ho, gradient_ih

def hold_out(train_im, train_lab, percent):
    num_hold_out = int(np.round(1/float(percent) * len(train_im)))
    hold_out_im = train_im[-num_hold_out:]
    hold_out_labels = train_lab[-num_hold_out:]
    train_im = train_im[:-num_hold_out]
    train_lab = train_lab[:-num_hold_out]
    return hold_out_im, hold_out_labels, train_im, train_lab

def num_approx(w_ih, w_ho, train_im, epsilon = .00001):
    epsilon_ih = epsilon*np.ones(w_ih.shape)
    epsilon_ho = epsilon*np.ones(w_ho.shape)

    #we will compute E_add first
    E_add_ih,E_add_ho = forward(train_im, w_ih + epsilon_ih, w_ho + epsilon_ho)
    #E_sub
    E_sub_ih, E_sub_ho = forward(train_im, w_ih - epsilon_ih, w_ho - epsilon_ho)
    #compute approximation (make this a function)
    num_approx_ih = numerical_approx_equation(E_add_ih - E_sub_ih)
    num_approx_ho = numerical_approx_equation(E_add_ho - E_sub_ho)

    return num_approx_ih, num_approx_ho

def numerical_approx_equation(E_plus, E_minus, epsilon = .00001):
    return (E_plus - E_minus) / (2 * (epsilon*np.ones(E_plus.shape)))

def gradient_checker(num_approx, grad_back):
    error = num_approx - grad_back
    if error < .0000000001:
        return "It works"
    else:
        return "Check your code"


#create minibatch
input_mini, label_mini = minibatch(train_dat, train_lab, 0, 128)
cat_mini = one_hot_encoding(label_mini)
input_bias = add_bias_term(input_mini)

#forward and backwards prop
final_ho, final_ih = forward(input_bias,w_ih, w_ho)
w_ho, w_ih, grad_ho, grad_ih = backprop(input_mini,cat_mini,final_ho, final_ih, w_ho, w_ih)

#gradient gradient_checker
#approx_ih, approx_ho = num_approx(w_ih, w_ho, input_bias, epsilon = .00001)
#check_ih = gradient_checker(approx_ih, grad_ih)
#check_ih = gradient_checker(approx_ho, grad_ho)

#1. add learning rate
#2. make epochs
#3. do gradient checker (correct? make script to check)
#4. make into a class
#5. fix the weights so that they are 64
#6. make w_ij and w_jk functions
#7. implement holdout
