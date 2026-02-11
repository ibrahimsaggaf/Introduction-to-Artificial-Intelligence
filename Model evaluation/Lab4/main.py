'''
Author: Dr Ibrahim Alsaggaf
Learning type: Supervised learning
Task: Multi-class classification
Dataset: The MNIST dataset of handwritten digit images [1]
Libraries:
    Scikit-learn [2]
    Pytorch [3]
    Matplotlib [4]

Model: MLP (Neural Network)

[1] Deng, L. (2012). The mnist database of handwritten digit images for machine learning research [best of the web]. 
    IEEE signal processing magazine, 29(6), 141-142.
[2] Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
[3] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). 
    Pytorch: An imperative style, high-performance deep learning library. Advances in neural information 
    processing systems, 32.
[4] Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in science & engineering, 9(03), 90-95.
'''

import time
import torch
from matplotlib import pyplot as plt

from model import Model
from utils import read_data

# Load the MNIST dataset
train_X, train_y = read_data('train_mnist.csv')
test_X, test_y = read_data('test_mnist.csv')
number_of_classes = torch.unique(train_y).size(0)

# Create a small instance of the model with only 1 hidden layer
small_model = Model(
    input_dimension=784,
    hidden_dim=256, 
    output_dimension=number_of_classes,  
    number_of_hidden=1,
    batch_size=64, 
    number_of_epochs=20,
    size='Small'
)

# Train and evaluate the small model
small_start = time.time()
small_model.fit(train_X, test_X, train_y, test_y)
small_end = time.time()
print(f'Small runtime: {(small_end - small_start) / 60:.4f} minutes')

# Create a large instance of the model with 4 hidden layer
large_model = Model(
    input_dimension=784,
    hidden_dim=256, 
    output_dimension=number_of_classes,  
    number_of_hidden=4,
    batch_size=64, 
    number_of_epochs=20,
    size='Large'
)

# Train and evaluate the large model
large_start = time.time()
large_model.fit(train_X, test_X, train_y, test_y)
large_end = time.time()
print(f'Large runtime: {(large_end - large_start) / 60:.4f} minutes')

# Plot both training and testing losses and save the figure
plt.plot(small_model.train_loss, color='blue', label='Train loss (Small)')
plt.plot(small_model.test_loss, color='red', label='Test loss (Small)')

plt.plot(large_model.train_loss, color='green', label='Train loss (Large)')
plt.plot(large_model.test_loss, color='orange', label='Test loss (Large)')

plt.title('Learning curves')
plt.xticks(range(0, 22, 2), range(0, 22, 2))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig('learning_curve.jpg', dpi=300, bbox_inches='tight')
plt.close()