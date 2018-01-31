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

def fan_in(inputs):
    return 1/(inputs**(1/2.0))

#minibatch of 128
def minibatch(train_set, train_labs, n, batch_size):
    batch = train_set[n*batch_size : (n*batch_size + batch_size), :]
    labels = train_labs[n*batch_size : (n*batch_size + batch_size)]
    return batch,labels

def rand_minibatch(train_set, train_labs, batch_size):
    index_rand = random.sample(xrange(len(train_set)), batch_size)
    batch = [train_set[n] for n in index_rand]
    labels = [train_labs[n] for n in index_rand]
    return batch,labels

def activation(x, w):
    return np.dot(x, w)

def sigmoid(x, derivative = False):
    y = 1 / (1 + np.exp(-1 * x))
    if (derivative == True):
        print 'derivative'
        return y*(1-y)
    else:
        print 'Normal Sigmoid'
        return y

def hyperbolic_tangent(x, derivative = False):
    if (derivative == True):
        return 1.14393/(np.cosh((2/3) * x) ** 2)
    else:
        return 1.7159 * np.tanh((2/3) * x)

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
def forward_ih(input_batch, w_input_hidden, bias_h, derivative=False):
    #input to hidden
    a_j = activation(input_batch, w_input_hidden) + bias_h  # Weighted sum of inputs + b
    z_i = sigmoid(a_j, derivative)                          # Activation Function
    return z_i

def forward_ho(hidden_activations, w_hidden_output, bias_o):
    #hidden to output
    a_k = activation(hidden_activations, w_hidden_output) + bias_o  # Weighted sum of inputs
    y_k = softmax(a_k)                                              # Activation Function
    return y_k

# BackProp: Output to hidden
def backprop_oh(w_jk, bias_o, x_h, l, lr):
    y = forward_ho(x_h, w_jk, bias_o)              # Recalculate classification probs
    t = one_hot_encoding(l)                # One-hot encode labels
    d_k = -delta_k(y,t)
    d_Ejk = np.dot(np.transpose(x_h), d_k)

    w_jk = w_jk + lr * d_Ejk                # Update weights
    bias_o = bias_o + lr * d_k              # Update bias
    return w_jk, bias_o, d_k

# Delta_K for output units
def delta_k(y, t):
    return (t - y)

# BackProp: Hidden to Input
def backprop_hi(w_ih, w_ho, bias_h, x, d_k, lr):
    g_h_der = forward_ih(batch_i, w_ih, bias_h, derivative=True)             # g'(a_j)
    d_j = - np.transpose(g_h_der) * (np.dot(w_ho, np.transpose(d_k))) # d_j =  g'(a_j) * sum(wjk * d_k)
    d_Eij = np.transpose( np.dot(d_j, x) )                         # -dEij = d_j * x_i
    d_Ebias = np.dot(d_j, np.transpose(np.ones(d_j.shape[1])))

    print d_Ebias.shape

    w_ih = w_ih + lr * d_Eij
    bias_h = bias_h + lr * d_Ebias
    return w_ih, bias_h

def hold_out(train_im, train_lab, percent):
    num_hold_out = int(np.round(1/float(percent) * len(train_im)))
    hold_out_im = train_im[-num_hold_out:]
    hold_out_labels = train_lab[-num_hold_out:]
    train_im = train_im[:-num_hold_out]
    train_lab = train_lab[:-num_hold_out]
    return hold_out_im, hold_out_labels, train_im, train_lab

def num_approx_ih(w_ih, train_im, epsilon = .00001):
    epsilon_v = epsilon*np.ones(w_ih.shape)
    E_add = forward_ih(train_im, w_ih + epsilon_v)
    E_sub = forward_ih(train_im, w_ih - epsilon_v)
    #compute approximation
    num_approx_ih = numerical_approx_equation(E_add, E_sub)

    return num_approx_ih

def num_approx_ho(g_h_b, w_ho, epsilon = .00001):
    epsilon_v = epsilon*np.ones(w_ho.shape)
    E_add = forward_ho(g_h_b, w_ho + epsilon_v)
    E_sub = forward_ho(g_h_b, w_ho - epsilon_v)
    #compute approximation
    num_approx_ho = numerical_approx_equation(E_add, E_sub)

    return num_approx_ho

def numerical_approx_equation(E_plus, E_minus, epsilon = .00001):
    return (E_plus - E_minus) / (2 * (epsilon*np.ones(E_plus.shape)))

def gradient_checker(num_approx, grad_back):
    error = abs(num_approx - grad_back)
    if error < .0000000001:
        return "It works"
    else:
        return "Check your code"

##############################################
# IMPLEMENTATION:

# PARAMETERS:
num_train = 60000    # Load num_train images
num_test = 10000      # Load num_test images
lr = 0.00000000001   # Learning rate
mu, sigma = 0, 0.1   # Parameters of Gaussian to initialize weights

batch_size = 128

num_input_units = 784   # Units in the imput layer
num_hidden_units = 64   # Units in the hidden layer
num_outputs = 10        # Units in the output layer

num_iterations = 100

# 1. Load Data
tr_i, tr_l, test_i, test_l = load_data(num_train, num_test)

# 2. Z-score data
tr_i, test_i = z_score_data(tr_i, test_i)

# mnist_printer(tr_i[0])

# 3. Initialize weights
w_ih = np.random.normal(mu, sigma, (num_input_units,num_hidden_units))
w_ho = np.random.normal(mu, sigma, (num_hidden_units,num_outputs))
b_h = np.random.normal(mu, sigma, num_hidden_units)
b_o = np.random.normal(mu, sigma, num_outputs)


# 4. TRAIN
etr = []
for it in xrange(0,num_iterations):
    print it
# Create minibatch
    batch_i, batch_l = minibatch(tr_i, tr_l, it, batch_size)
    # batch_i = add_bias_term(batch_i)

    #For_Prop: input to hidden
    z_i = forward_ih( batch_i, w_ih, b_h) # Activation Function of hidden units
    #For_Prop: hidden to output
    y_k = forward_ho(z_i, w_ho, b_o) # Activation Function of output units

    #Backprop: Output to hidden:
    lr = 0.0001
    w_ho, b_o, d_k = backprop_oh(w_ho, b_o, z_i, batch_l, lr) # Update w_jk weights + b_o
    #Backprop: Hidden to input:
    w_ih, b_h = backprop_hi(w_ih, w_ho, b_h, batch_i, d_k, lr) # Update w_ij weights + b_h


    #Calculate error
    z_i = forward_ih(batch_i, w_ih, b_h)
    y_k = forward_ho(z_i, w_ho, b_o)

    pred_tr = np.argmax(y_k, 1)
    error_tr = 100.0 * (1 - 1.0*(np.sum(pred_tr != tr_l)) / (1.0 * tr_i.shape[0]))
    etr.append(error_tr)


# Plot Error
plt.figure()
plt.plot(etr, label='Training Data, (Classification Accuracy) = %.2f%s' %(np.max(etr), '%'))
# plt.plot(eh, label='Hold-out Data, (Classification Accuracy) = %.2f%s' %(np.max(eh), '%'))
# plt.plot(ete, label='Testing Data, (Classification Accuracy) = %.2f%s' %(np.max(ete), '%'))
plt.title('Percent correct classification: SOFTMAX')
plt.xlabel('# Epochs')
plt.ylabel('Percent Correct Classification')
plt.legend(loc='lower right')
plt.show(block=False)


#Tips and Tricks
#1. Random Sampling
batch_i, batch_l = rand_minibatch(tr_i, tr_l, batch_size)

#2. sigmoid in Section 4.4
#in forward_ih change sigmoid to hyperbolic_tangent(a_j)

#3. Initialize the input weights to each unit using a distribution with 0 mean and standard deviation 1/sqrt(fan-in), where the fan-in is the number of inputs to the unit.
mu = 0
w_ih = np.random.normal(mu, fan_in(num_input_units), (num_input_units,num_hidden_units))
w_ho = np.random.normal(mu, fan_in(num_hidden_units), (num_hidden_units,num_outputs))

#4 Use momentum, with an alpha of 0.9.
#create an array to store prev delta values before for loop
prev_delta_jk = []
alpha = 0.9
gradient_delta = lr * d_Ejk
w_ho += gradient_delta + (alpha * prev_delta_jk[-1]))
prev_delta_jk.append(gradient_delta)

prev_delta_ij = []
gradient_delta = lr * d_Eij
w_ih += gradient_delta + (alpha * prev_delta_ij[-1]))
prev_delta_ij.append(gradient_delta)



#gradient gradient_checker (check this)
approx_ih = num_approx_ih(w_ih, batch_i_b)
approx_ho = num_approx_ho(z_i_b, w_ho)

# Backwards prop
#w_ho, w_ih, grad_ho, grad_ih = backprop(batch_i,batch_1h_l,y_k, z_i, w_ho, w_ih)

error_ih = gradient_checker(approx_ih, grad_ih)
error_ho = gradient_checker(approx_ho, grad_ho)


#3. make into a class
#4. implement holdout
