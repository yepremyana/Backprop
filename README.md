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

The MNIST dataset is already unploaded into this repository. 

## Running the tests

## Contributors
Pablo tostado and Alice Yepremyan
