# Using Backpropagation to classify MNIST dataset.

Backpropagation is a supervised neural network commonly used in the field of Machine Learning which utilizes gradient descent. Backpropagation is typically used on neural networks that contain hidden layers. Herein, we built a 3-layer Neural network using Backpropagation with sigma and Softmax activations for image classification of the MNIST dataset. Training of the algorithms was done using gradient descent to minimize the cross entropy loss function. Furthermore, we implemented 'hold-out' testing, 'annealing' and 'early stopping' in order to avoid overfitting and maximize performance. Finally, we improved our model using various tips and tricks. We achieved a classification accuracy of \% for the initial Backpropagation and of \% for the optimized Backpropagation model on handwritten digits. 

## Getting Started
### Prerequisites
To run this code the following packages to be imported
```
from mnist import MNIST
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pdb
import copy
```

The MNIST dataset is already uploaded into this repository. 

## Neural Network Structure
The network for this project consisted of 3 layers. The input layer was composed of 784 nodes corresponding to the 784 pixels of the input images, plus an extra node related to the bias of the unit. The bias unit is initialized at 1. The input layer was connected to a hidden layer with 64 hidden nodes. The connections, or weights, between these layers were set to a Gaussian distribution with $\mu$ = 0 and $\sigma$ = 0.1. A bias was also added to this layer. The final layer consisted of 10 output nodes corresponding to labels 0-9. The sigmoid activation will be used from the input layer to the hidden layer. And softmax will be applied to the output layer.

## Running the tests
Output of backprop.py file will include the raw model without any propagation as well as figures after optimizations were performed on our model

### Running model
```
python backprop.py 
```
The tricks included are:
1. shuffling of images
2. utilizing fan-in for the initial weight distribution
3. momentum where alpha = 0.9
4. using the hyperbolic tangent function instead of the sigmoid

Figure titles will be labeled with the tricks used as well as a final image with the optmized model.

Please refer to the report attached to this submission for the detailed explaination of the backprop model.

## Contributors
Pablo tostado and Alice Yepremyan
