from mnist import MNIST
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pdb
import copy


#<editor-fold desc="Print Weights">
#Prints Image. Input: 1D array 784
def mnist_printer(mnist_array, save=False):
    pixmap = weights_to_2d(mnist_array).astype(float)
    # print pixmap.shape #28x28
    plt.imshow(pixmap, cmap=cm.gray, interpolation='nearest')
    plt.show(block=False)

# takes 1D array turns it into 2D arrays. 784 weights to 28x28
def weights_to_2d(weights):
    dim1 = int(np.sqrt(len(weights)))
    dim2 = int(len(weights) / dim1)
    weights = weights[:dim1*dim2] # This is for adding the occlusions.
    return copy.deepcopy(np.reshape(weights, (dim1, dim2)))
#</editor-fold>

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
    batch = train_set[n*batch_size : (n*batch_size + batch_size), :]
    labels = train_labs[n*batch_size : (n*batch_size + batch_size)]
    return batch,labels

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
    return np.array([np.append(1,x) for x in x_array])

def one_hot_encoding(label_to_1hotencode):
    encoded_list = list()
    for label in label_to_1hotencode:
        label_zero = [0 for _ in xrange(10)]
        label_zero[label] = 1
        encoded_list.append(label_zero)
    return np.array(encoded_list)

def softmax(activation_k):
    exp_ak = np.exp(activation_k)  # Exp of my class
    sum_exp_ak = np.sum(exp_ak, 1) # Sum of exp of classes
    sum_exp_ak = np.reshape(sum_exp_ak, (exp_ak.shape[0], 1))
    sum_exp_ak = np.repeat(sum_exp_ak, exp_ak.shape[1], axis=1)
    return exp_ak / (1.0 * sum_exp_ak) # Normalized outputs of classifier

#feedforward
def forward_ih(input_batch, w_input_hidden):
    #input to hidden
    a_j = activation(input_batch, w_input_hidden) # Weighted sum of inputs
    g_h = sigmoid(a_j)                            # Activation Function
    return g_h

def forward_ho(hidden_activations, w_hidden_output):
    #hidden to output
    a_k = activation(hidden_activations, w_hidden_output) # Weighted sum of inputs
    g_o = softmax(a_k)                               # Activation Function
    return g_o

# BackProp: Output to hidden
def backprop_oh(w_jk, x_h, l, lr):
    y = forward_ho(x_h, w_jk)              # Recalculate classification probs
    t = one_hot_encoding(l)                # One-hot encode labels
    d_E = -np.dot(np.transpose(x), delta_k(y,t))
    return w_jk + lr * d_E  # Update weights

# Delta_K for output units
def delta_k(y, t):
    return (t - y)

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
    E_add_ih, _ = forward(train_im, w_ih + epsilon_ih, w_ho)
    E_sub_ih, _ = forward(train_im, w_ih - epsilon_ih, w_ho)

    _, E_add_ho = forward(train_im, w_ih, w_ho + epsilon_ho)
    _, E_sub_ho = forward(train_im, w_ih, w_ho + epsilon_ho)

    #compute approximation (make this a function)
    num_approx_ih = numerical_approx_equation(E_add_ih, E_sub_ih)
    num_approx_ho = numerical_approx_equation(E_add_ho, E_sub_ho)

    return num_approx_ih, num_approx_ho

def numerical_approx_equation(E_plus, E_minus, epsilon = .00001):
    return (E_plus - E_minus) / (2 * (epsilon*np.ones(E_plus.shape)))

def gradient_checker(num_approx, grad_back):
    error = num_approx - grad_back
    if error < .0000000001:
        return "It works"
    else:
        return "Check your code"

##############################################
# IMPLEMENTATION:

# 1. Load Data
num_train = 10000
num_test = 1000
tr_i, tr_l, test_i, test_l = load_data(num_train, num_test)

# 2. Z-score data
tr_i, test_i = z_score_data(tr_i, test_i)

# mnist_printer(tr_i[0])

# Initialize weights
mu, sigma = 0, 0.1
w_ih = np.random.normal(mu, sigma, (785,64))
w_ho = np.random.normal(mu, sigma, (65,10))

# Create minibatch
batch_i, batch_l = minibatch(tr_i, tr_l, 0, 2)
# batch_1h_l = one_hot_encoding(batch_l)
batch_i = add_bias_term(batch_i)

#For_Prop: input to hidden
g_h = forward_ih(batch_i,w_ih) # Activation Function of hidden units
g_h = add_bias_term(g_h)       # Add bias before passing Activations to output layer
#For_Prop: hidden to output
g_o = forward_ho(g_h, w_ho) # Activation Function of output units

#Backprop: Output to hidden:
lr = 0.0001
w_ho = backprop_oh(w_ho, g_h, batch_l, lr) # Update w_jk weights
#Backprop: Hidden to input:




# Backwards prop
w_ho, w_ih, grad_ho, grad_ih = backprop(batch_i, batch_1h_l, final_ho, final_ih, w_ho, w_ih)

#where t is the expected and y is the output (from forwards_prop)
delta_k = batch_1h_l - final_ho
w_ho = w_ho[1:]
w_ih = w_ih[1:]

#w_ij
activation_ih = activation(batch_i, w_ih)
error = sigmoid(activation_ih, derivative = True)
c = error * np.dot(delta_k, w_ho.T)
gradient_ih = np.dot(batch_i.T, c)
w_ih += gradient_ih

#w_jk
z = softmax(final_ih)
gradient_ho = np.dot(z.T, delta_k)
w_ho += gradient_ho

return w_ho, w_ih, gradient_ho, gradient_ih




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
