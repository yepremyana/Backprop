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
    batch = train_set[n*batch_size : (n*batch_size + batch_size),:]
    labels = train_labs[n*batch_size : (n*batch_size + batch_size)]
    return batch,labels

def rand_minibatch(train_set, train_labs, batch_size):
    index_rand = random.sample(xrange(len(train_set)), batch_size)
    batch = [train_set[n] for n in index_rand]
    labels = [train_labs[n] for n in index_rand]
    return np.array(batch),np.array(labels)

def activation(x, w):
    return np.dot(x, w)

def sigmoid(x, derivative=False):
    y = 1 / (1 + np.exp(-1 * x))
    if (derivative == True):
        # print 'derivative'
        return y*(1-y)
    else:
        # print 'Normal Sigmoid'
        return y

def hyperbolic_tangent(x, derivative=False):
    if (derivative == True):
        return 1.7159 * 2.0 * (1.0 - (np.tanh(2.0*x/3.0)**2))/3.0
    else:
        return 1.7159 * np.tanh((2.0*x/3.0))

#Add a 1 in front of every input vector that accounts for the bias weight
def add_bias_term(x_array):
    a = np.array(x_array)
    x_bias = np.insert(a, 0, 1, axis=1)
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
def forward_ih(input_batch, w_input_hidden, derivative=False):
    #input to hidden
    a_j = activation(input_batch, w_input_hidden)   # Weighted sum of inputs
    z_j = sigmoid(a_j, derivative)                          # Activation Function
    return z_j

def forward_ho(hidden_activations, w_hidden_output):#, bias_o):
    #hidden to output
    a_k = activation(hidden_activations, w_hidden_output) #+ bias_o  # Weighted sum of inputs
    y_k = softmax(a_k)                                              # Activation Function
    return y_k

# BackProp: Output to hiddend
def backprop(z_j, d_k, lr):
    d_Ejk = lr * np.dot(z_j, d_k)
    return d_Ejk

def get_dk_gradient(w_jk, z_j, l):
        y = forward_ho(z_j, w_jk)              # Recalculate classification probs
        t = one_hot_encoding(l)                # One-hot encode labels
        return delta_k(y,t)

# Delta_K for output units
def delta_k(y, t):
    return (t - y)

def get_dj_gradient(w_ij, w_jk, x, d_prev):
        g_h_der = forward_ih(x, w_ij, derivative=True)             # g'(a_j)
        # d_j =  g'(a_j) * sum(wjk * d_k) !!!!! THIS IS THE CRITICAL STEP, we do not take the row of Wjk corresponding to their biases
        d_j = np.transpose(g_h_der) * (np.dot(w_jk[1:,:], np.transpose(d_prev)))
        print g_h_der.shape
        return d_j

#Extract a hold-out set of x% from the training data:
def hold_out(train_im, train_lab, num_hold_out):
    hold_out_im = train_im[-num_hold_out:]
    hold_out_labels = train_lab[-num_hold_out:]
    train_im = train_im[:-num_hold_out]
    train_lab = train_lab[:-num_hold_out]
    return np.array(hold_out_im), np.array(hold_out_labels), \
           np.array(train_im), np.array(train_lab)

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

def get_prediction_error(input_i, input_l, w_ih, w_h1, w_ho):
    z_j = forward_ih(input_i, w_ih)    # Input to h1
    z_j = add_bias_term(z_j)
    z_h = forward_ih(z_j, w_h1)         # Input to h2
    z_h = add_bias_term(z_h)
    y_k = forward_ho(z_h, w_ho)         # Activation Function of output units

    pred_l = np.argmax(y_k, 1)   #Predicted labels
    return 100.0 * (np.sum(pred_l == input_l)) / (1.0 * input_l.shape[0]) #Return ACCURACY

##############################################
# IMPLEMENTATION:

# PARAMETERS:
num_train = 60000    # Load num_train images
num_test = 10000      # Load num_test images
num_hold_out = 10000  # How many images from training used for validation

lr = 0.01 # Learning rate
mu, sigma = 0, 0.1   # Parameters of Gaussian to initialize weights

batch_size = 128

num_input_units = 784   # Units in the imput layer
num_h1 = 58   # Units in the hidden layer
num_h2 = 58
num_outputs = 10        # Units in the output layer

# 1. Load Data
tr_i, tr_l, test_i, test_l = load_data(num_train, num_test)
# 2. Z-score data
tr_i, test_i = z_score_data(tr_i, test_i)
# 3. Split training data into training and validation (hold-out set):
hi, hl, tr_i, tr_l = hold_out(tr_i, tr_l, num_hold_out)

# Add extra 1st column to input images for biases
tr_i = add_bias_term(tr_i)
hi = add_bias_term(hi)
test_i = add_bias_term(test_i)

# 3. Initialize weights
w_ih = np.random.normal(mu, sigma, (num_input_units+1, num_h1)) #+1 for bias
w_h1 = np.random.normal(mu, sigma, (num_h1+1, num_h1))     #+1 for bias
w_ho = np.random.normal(mu, sigma, (num_h2+1, num_outputs))     #+1 for bias


# 4. TRAIN
tr_acc = []
val_acc = []
test_acc = []
for epochs in xrange (1,30):

    num_iterations = int(tr_i.shape[0] / 128.0)
    # acc = []
    for it in xrange(0,num_iterations):
        # Create minibatch
        batch_i, batch_l = minibatch(tr_i, tr_l, it, batch_size)

        #For_Prop: Input to hidden
        z_j = forward_ih( batch_i, w_ih)    # Input to h1
        z_j = add_bias_term(z_j)
        z_h = forward_ih(z_j, w_h1)         # Input to h2
        z_h = add_bias_term(z_h)
        y_k = forward_ho(z_h, w_ho)         # Activation Function of output units

        # BACKPROP:
        d_k = get_dk_gradient(w_ho, z_h, batch_l)
        d_h1 = get_dj_gradient(w_h1, w_ho, z_j, d_k)
        d_j = get_dj_gradient(w_ih, w_h1, batch_i, d_h1.T)

        w_ih += backprop(batch_i.T, d_j.T, lr)
        w_h1 += backprop(z_j.T, d_h1.T, lr)
        w_ho += backprop(z_h.T, d_k, lr)

    #Save training, validation & testing errors:
    tr_acc.append(get_prediction_error(tr_i, tr_l, w_ih, w_h1, w_ho))
    val_acc.append(get_prediction_error(hi, hl, w_ih, w_h1, w_ho))
    test_acc.append(get_prediction_error(test_i, test_l, w_ih, w_h1, w_ho))
    
# Plot Error
plt.figure()
plt.plot(tr_acc, label='Training Data, (Training Accuracy) = %.2f%s' %(np.max(tr_acc), '%'))
plt.plot(val_acc, label='Hold-Out Data, (Validation Accuracy) = %.2f%s' %(np.max(val_acc), '%'))
plt.plot(test_acc, label='Testing Data, (Testing Accuracy) = %.2f%s' %(np.max(test_acc), '%'))
plt.title('Percent correct classification: SOFTMAX')
plt.xlabel('# Epochs')
plt.ylabel('Percent Correct Classification')
plt.legend(loc='lower right')
plt.show(block=False)
