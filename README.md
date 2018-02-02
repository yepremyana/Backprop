# Using Backpropagation to classify MNIST dataset.

Backpropagation is a supervised neural network commonly used in the field of Machine Learning which utilizes gradient descent. Backpropagation is typically used on neural networks that contain hidden layers. Herein, we built a 3-layer Neural network using Backpropagation with sigma and Softmax activations for image classification of the MNIST dataset. Training of the algorithms was done using gradient descent to minimize the cross entropy loss function. Furthermore, we implemented 'hold-out' testing, 'annealing' and 'early stopping' in order to avoid overfitting and maximize performance. Finally, we improved our model using various tips and tricks. This achieved a classification accuracy of 96.62\% for the initial basic architecture and of 97.76\% for the optimized model including 'training tricks' on handwritten digits. 

## Getting Started

The MNIST dataset is already uploaded into this repository. 

## Neural Network Structure
The network for this project consisted of 3 layers. The input layer was composed of 784 nodes corresponding to the 784 pixels of the input images, plus an extra node related to the bias of the unit. The bias unit is initialized at 1. The input layer was connected to a hidden layer with 64 hidden nodes. The connections, or weights, between these layers were set to a Gaussian distribution with $\mu$ = 0 and $\sigma$ = 0.1. A bias was also added to this layer. The final layer consisted of 10 output nodes corresponding to labels 0-9. The sigmoid activation will be used from the input layer to the hidden layer. And softmax will be applied to the output layer.

The code contains all the necessary functions to import the MNIST data, train the neural network using backpropagation, and plot the accuracy and loss function.

## Running the tests
Output of backprop.py file will include the raw model without any propagation as well as figures after optimizations were performed on our model

### Running model
```
python backprop.py 
```

######  To change by the user:

The user may play with the network architecture by changing the parameters found in the section 'PARAMETERS' after the functions section. The parameters that can be tunned are:

- Num training / holdout / testing samples
- Learning rate (lr)
-Paramters of Gaussian to initialize weights
-Batch size
-Number of units in each layer
-Early stopping (bool)

That's it, hit play and enjoy!


#####
The tricks included are:
1. shuffling of images
2. utilizing fan-in for the initial weight distribution
3. momentum where alpha = 0.9
4. using the hyperbolic tangent function instead of the sigmoid

Figure titles will be labeled with the tricks used as well as a final image with the optmized model.
### Figures expected from running the model
1. Initial raw model
2. shuffling of images
3. momentum where alpha = 0.9
4. using the hyperbolic tangent function instead of the sigmoid
5. doubling hidden layers
6. adding a second layer
7. most optimized model

Please refer to the report attached to this submission for the detailed explaination of the backprop model.

## Contributors
Pablo tostado and Alice Yepremyan
