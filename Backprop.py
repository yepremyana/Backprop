from mnist import MNIST
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pdb
import copy


#<editor-fold desc="Print Weights">
def mnist_printer(mnist_array, save=False):
    '''
    Prints Image. Input: 1D array 784
    '''
    pixmap = weights_to_2d(mnist_array).astype(float)
    # print pixmap.shape #28x28
    plt.imshow(pixmap, cmap=cm.gray, interpolation='nearest')
    plt.show(block=False)

def weights_to_2d(weights):
    '''
    takes 1D array turns it into 2D arrays. 784 weights to 28x28
    '''
    dim1 = int(np.sqrt(len(weights)))
    dim2 = int(len(weights) / dim1)
    weights = weights[:dim1*dim2] # This is for adding the occlusions.
    return copy.deepcopy(np.reshape(weights, (dim1, dim2)))
#</editor-fold>

def load_data(num_train, num_test, directory='mnist'):
    '''
    Will load MNIST data and return the first and last specified number of training & testing images respectively
    '''
    mndata = MNIST(directory)
    train_dat, train_lab = mndata.load_training()
    test_dat, test_lab = mndata.load_testing()
    return np.array(train_dat[:num_train]), np.array(train_lab[:num_train]), \
           np.array(test_dat[-num_test:]), np.array(test_lab[-num_test:])

def z_score_data(train_dat, test_dat):
    '''
    Images /127.5 - 1 so that they are in range [-1,1]
    '''
    train_dat = train_dat/127.5 -1
    test_dat = test_dat/127.5 - 1
    return train_dat, test_dat

def fan_in(inputs):
    '''
    Calculates sigma for initializing the weights of the network
    '''
    return 1/(inputs**(1/2.0))

def minibatch(train_set, train_labs, n, batch_size):
    '''
    Creates minibatches from a larger dataset while retaining label values
    '''
    batch = train_set[n*batch_size : (n*batch_size + batch_size),:]
    labels = train_labs[n*batch_size : (n*batch_size + batch_size)]
    return batch,labels

def rand_minibatch(train_set, train_labs, batch_size):
    '''
    randomizes order of samples
    '''
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

def add_bias_term(x_array):
    '''
    Add a 1 in front of every input vector that accounts for the bias weight
    '''
    a = np.array(x_array)
    x_bias = np.insert(a, 0, 1, axis=1)
    return np.array([np.append(1,x) for x in x_array])

def one_hot_encoding(label_to_1hotencode):
    '''
    Makes labels into vectors
    '''
    encoded_list = list()
    for label in label_to_1hotencode:
        label_zero = [0 for i in xrange(10)]
        label_zero[label] = 1
        encoded_list.append(label_zero)
    return np.array(encoded_list)

def softmax(activation_k):
    exp_ak = np.exp(activation_k)  # Exp of my class
    sum_exp_ak = np.sum(exp_ak, 1) # Sum of exp of classes
    sum_exp_ak = np.reshape(sum_exp_ak, (exp_ak.shape[0], 1))
    sum_exp_ak = np.repeat(sum_exp_ak, exp_ak.shape[1], axis=1)
    return exp_ak / (1.0 * sum_exp_ak) # Normalized outputs of classifier

def forward_ih(input_batch, w_input_hidden, derivative=False):
    '''
    The forward propagation between the input layer and the hidden layer with
    either a sigmoid or hyperbolic tangent sigmoid.
    '''
    #input to hidden
    a_j = activation(input_batch, w_input_hidden)   # Weighted sum of inputs
    #part 4b
    z_j = hyperbolic_tangent(a_j, derivative)

    #z_j = sigmoid(a_j, derivative)                          # Activation Function
    return z_j

def forward_ho(hidden_activations, w_hidden_output):
    '''
    The forward propagation between the hidden layer and the output layer with
    softmax applied to the output layer.
    '''
    #hidden to output
    a_k = activation(hidden_activations, w_hidden_output) #+ bias_o  # Weighted sum of inputs
    y_k = softmax(a_k)                                              # Activation Function
    return y_k

def backprop_oh(w_jk, z_j, d_k, lr):
    '''
    Backpropagation between the output layer and the hidden layer. Function
    returns the update for w_jk.
    '''
    d_Ejk = np.dot(np.transpose(z_j), d_k)
    w_jk_update = lr * d_Ejk                # Update weights
    return w_jk_update

def get_dk_gradient(w_jk, z_j, l):
    y = forward_ho(z_j, w_jk)              # Recalculate classification probs
    t = one_hot_encoding(l)                # One-hot encode labels
    return delta_k(y,t)

def delta_k(y, t):
    '''
    Delta_K for output units
    '''
    return (t - y)

def backprop_hi(w_ih, w_ho, x, d_k, lr):
    '''
    Backpropagation between the hidden layer and the input layer. Function
    returns the update for w_ij.
    '''
    g_h_der = forward_ih(batch_i, w_ih, derivative=True)             # g'(a_j)
    d_j = np.transpose(g_h_der) * (np.dot(w_ho[1:,:], np.transpose(d_k)))
    d_Eij = np.transpose( np.dot(d_j, x) )                         # -dEij = d_j * x_i

    w_ih_update = lr * d_Eij
    return w_ih_update

def hold_out(train_im, train_lab, num_hold_out):
    '''
    Extract a hold-out set of percentage from the training data. Used to stop training before overfitting
    '''
    hold_out_im = train_im[-num_hold_out:]
    hold_out_labels = train_lab[-num_hold_out:]
    train_im = train_im[:-num_hold_out]
    train_lab = train_lab[:-num_hold_out]
    return np.array(hold_out_im), np.array(hold_out_labels), \
           np.array(train_im), np.array(train_lab)

def num_approx_ih(w_ih, w_ho, train_im, label, it,ip, epsilon = .01):
    '''
    Calculation of the numerical approximation and the backpropagation of the
    gradient w_ij
    '''
    z_j = forward_ih(train_im, w_ih)
    z_j = add_bias_term(z_j)
    delta_k = get_dk_gradient(w_ho, z_j, label)
    w_ih_update = backprop_hi(w_ih, w_ho, train_im, delta_k, .01)

    w_ih[it][ip] = w_ih[it][ip] + epsilon
    E_plus = loss_funct(train_im, label, w_ih, w_ho)

    w_ih[it][ip] = w_ih[it][ip] - (2.0*epsilon)
    E_sub = loss_funct(train_im, label, w_ih, w_ho)

    num = numerical_approx_equation(E_plus, E_sub)
    return w_ih_update, num

def num_approx_ho(w_ih, w_ho, train_im, label, it, ip, epsilon = .01):
    '''
    Calculation of the numerical approximation and the backpropagation of the
    gradient w_jk
    '''
    z_j = forward_ih(train_im, w_ih)
    z_j = add_bias_term(z_j)
    delta_k = get_dk_gradient(w_ho, z_j, label)
    w_ho_update = backprop_oh(w_ho, z_j, delta_k, .01)

    w_ho[it][ip] = w_ho[it][ip] + epsilon
    E_plus = loss_funct(train_im, label, w_ih, w_ho)

    w_ho[it][ip] = w_ho[it][ip] - (2.0*epsilon)
    E_sub = loss_funct(train_im, label, w_ih, w_ho)

    num = numerical_approx_equation(E_plus, E_sub)
    return w_ho_update, num

def numerical_approx_equation(E_plus, E_minus, epsilon = .01):
    return (E_plus - E_minus) / (2.0 * (epsilon))

def grad_checker(num_approx, grad_back):
    '''
    Checks that the difference between the num and backprop calc is small.
    '''
    error = abs(num_approx - grad_back)
    if error < .0001:
        return "It works"
    else:
        return "Check your code"

def get_prediction_error(input_i, input_l, w_ih, w_ho):
    z_j = forward_ih(input_i, w_ih)
    z_j = add_bias_term(z_j)
    y_k = forward_ho(z_j, w_ho)
    pred_l = np.argmax(y_k, 1)   #Predicted labels
    return 100.0 * (np.sum(pred_l == input_l)) / (1.0 * input_l.shape[0]) #Return ACCURACY

def loss_funct(input_i, input_l, w_ih, w_ho):
    '''
    Calculates the cross entropy, the loss function.
    '''
    z_j = forward_ih(input_i, w_ih)
    z_j = add_bias_term(z_j)
    y = forward_ho(z_j, w_ho)
    t = one_hot_encoding(input_l)
    #Normalize w.r.t # training examples and #categories
    return (-1.0 / (input_i.shape[0] * w_ho.shape[1])) * (np.sum(t * np.log(y)))

def early_stopping(v_acc):
    '''
    Early stopping of training is implemented if the error goes up over 5 epochs
    '''
    if (all(v_acc[-1] < i for i in [v_acc[-2], v_acc[-3], v_acc[-4], v_acc[-5], v_acc[-6]])): return True

##############################################

# PARAMETERS:
num_train = 60000    # Load num_train images
num_test = 10000      # Load num_test images
num_hold_out = 10000  # How many images from training used for validation

lr = 0.0003  #Best: 0.01
mu, sigma = 0, 0.1   # Parameters of Gaussian to initialize weights
alpha = 0.9
batch_size = 128

num_input_units = 784   # Units in the imput layer
num_hidden_units = 128   # Units in the hidden layer
num_outputs = 10        # Units in the output layer

earlyStop = False       # Stop if validation error < 5 previous epochs

##############################################
# IMPLEMENTATION:

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
#w_ih = np.random.normal(mu, sigma, (num_input_units+1, num_hidden_units)) #+1 for bias
#w_ho = np.random.normal(mu, sigma, (num_hidden_units+1, num_outputs))     #+1 for bias

#4c Fan-in
mu = 0
w_ih = np.random.normal(mu, fan_in(num_input_units+1), (num_input_units+1,num_hidden_units))
w_ho = np.random.normal(mu, fan_in(num_hidden_units+1), (num_hidden_units+1,num_outputs))

# 4. TRAIN
tr_acc = []
val_acc = []
test_acc = []
Ltr = []
Lh = []
Lte = []

for epoch in xrange(40):
    print epoch

    num_iterations = int(tr_i.shape[0] / 128.0)
    acc = []
    #4a Shuffling the data
    tr_i, tr_l = rand_minibatch(tr_i, tr_l,50000)

    prev_update_jk = np.zeros(w_ho.shape)
    prev_update_ij = np.zeros(w_ih.shape)

    for it in xrange(0,num_iterations):
        # Create minibatch
        batch_i, batch_l = minibatch(tr_i, tr_l, it, batch_size)

        #For_Prop: Input to hidden
        z_j = forward_ih( batch_i, w_ih) # Activation Function of hidden units
        z_j = add_bias_term(z_j) # Add extra 1st column to hidden activations for biases
        #For_Prop: hidden to output
        y_k = forward_ho(z_j, w_ho) # Activation Function of output units

        # BACKPROP:
        d_k = get_dk_gradient(w_ho, z_j, batch_l)
        # 1st: hidden to input (Bc we need old w_jk)
        w_ih_update = backprop_hi(w_ih, w_ho, batch_i, d_k, lr) # Update w_ij weights
        #w_ih += w_ih_update

        #4d momentum
        w_ih += w_ih_update + (alpha * prev_update_ij)

        # 2nd: output to hidden
        w_ho_update = backprop_oh(w_ho, z_j, d_k, lr) # Update w_jk weights
        #w_ho += w_ho_update

        #4d momentum
        w_ho += w_ho_update + (alpha * prev_update_jk)
        prev_update_jk,prev_update_ij = w_ho_update, w_ih_update

    #Calculate Entropy
    Ltr.append(loss_funct(tr_i, tr_l, w_ih, w_ho))
    Lh.append(loss_funct(hi, hl, w_ih, w_ho))
    Lte.append(loss_funct(test_i, test_l, w_ih, w_ho))

    #Accuracies:
    #Save training, validation & testing errors:
    # tr_acc.append(np.mean(acc)) # Average of training error during training
    tr_acc.append(get_prediction_error(tr_i, tr_l, w_ih, w_ho))
    val_acc.append(get_prediction_error(hi, hl, w_ih, w_ho))
    test_acc.append(get_prediction_error(test_i, test_l, w_ih, w_ho))

    # Early stopping:
    if (earlyStop == True and epoch>7):
        if (early_stopping(val_acc) == True):
            print 'Early stopped'
            break

# Plot Error
plt.figure()
plt.plot(tr_acc, label='Training Data, (Training Accuracy) = %.2f%s' %(np.max(tr_acc), '%'))
plt.plot(val_acc, label='Hold-Out Data, (Validation Accuracy) = %.2f%s' %(np.max(val_acc), '%'))
plt.plot(test_acc, label='Testing Data, (Testing Accuracy) = %.2f%s' %(np.max(test_acc), '%'))
plt.title('Percent correct classification: Backpropagation and Shuffling')
plt.xlabel('# Epochs')
plt.ylabel('Percent Correct Classification')
plt.legend(loc='lower right')
plt.show(block=False)

# Plot Loss
plt.figure()
plt.plot(Ltr, label='Training Data')
plt.plot(Lh, label='Hold-Out Data')
plt.plot(Lte, label='Testing Data')
plt.title('Cross Entropy loss function')
plt.xlabel('# Epochs')
plt.ylabel('Cross Entropy')
plt.legend(loc='lower right')
plt.show(block=False)

#Graph for numerical approx
'''
bias_jk = []
bias_ij = []
num_ij = []
num_jk = []
#batch_i, batch_l = minibatch(tr_i, tr_l, 3, 1)
for i in xrange(10):
    batch_i, batch_l = minibatch(tr_i, tr_l, i+10, 1)
    exact_ij, grad_num_ij = num_approx_ih(w_ih,w_ho,batch_i,batch_l,i,i)
    exact_jk, grad_num_jk = num_approx_ho(w_ih,w_ho,batch_i,batch_l,i,i)
    exact_b_jk, grad_num_b_jk = num_approx_ho(w_ih,w_ho,batch_i,batch_l,0,i)
    exact_b_ij, grad_num_b_ij = num_approx_ih(w_ih,w_ho,batch_i,batch_l,0,i)
    num_ij.append(abs(grad_num_ij - exact_ij[i][i]))
    num_jk.append(abs(grad_num_jk - exact_jk[i][i]))
    bias_jk.append(abs(grad_num_b_jk - exact_b_jk[0][i]))
    bias_ij.append(abs(grad_num_b_ij - exact_b_ij[0][i]))

plt.figure()
plt.plot(num_ij,'ro', label='Changing weight in w_ij')
plt.plot(num_jk, 'bo', label='Changing weight in w_jk')
plt.plot(bias_ij, 'go',label='Changing weight in bias w_ij')
plt.plot(bias_jk, 'ko',label='Changing weight in bias w_jk')
plt.axhline(y=0.0001, color='r', linestyle='-')
plt.ylim(ymin=0)
plt.title('Gradient checker')
plt.xlabel('Examples')
plt.ylabel('difference between numerical approx and backprop')
plt.legend(loc='lower right')
plt.show(block=False)
'''
