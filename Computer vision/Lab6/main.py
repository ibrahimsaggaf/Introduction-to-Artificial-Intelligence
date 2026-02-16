'''
Author: Dr Ibrahim Alsaggaf
Learning type: Supervised learning
Task: Large-scale image classification
Dataset: Tiny ImageNet [1]
Libraries:
    datasets [2]
    pillow [3]
    numpy [4]
    Pytorch [5]
    Matplotlib [6]

Model: ResNet

[1] Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021). 
    Informer: Beyond efficient transformer for long sequence time-series forecasting. 
    In Proceedings of the AAAI conference on artificial intelligence (Vol. 35, No. 12, 
    pp. 11106-11115).
[2] Lhoest, Q., Del Moral, A. V., Jernite, Y., Thakur, A., Von Platen, P., Patil, S., ... & Wolf, T. (2021). 
    Datasets: A community library for natural language processing. In Proceedings of the 2021 conference on 
    empirical methods in natural language processing: system demonstrations (pp. 175-184).
[3] Lundh, F. and Clark, A. (2015). Pillow (PIL Fork) (Version 2.8.0) [Computer software]. 
    Available from https://python-pillow.github.io/
[4] Harris, C. R., Millman, K. J., Van Der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., ... & 
    Oliphant, T. E. (2020). Array programming with NumPy. nature, 585(7825), 357-362.
[5] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). 
    Pytorch: An imperative style, high-performance deep learning library. Advances in neural information 
    processing systems, 32.
[6] Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in science & engineering, 9(03), 90-95.
'''


import time
import torch
import numpy as np
from matplotlib import pyplot as plt

from model import Model
from utils import read_data

# Download and preprocess the Tiny ImageNet dataset
# Withot stratified sampling (all data points)
train_image, test_image, train_label, test_label = read_data('zh-plus/tiny-imagenet')

# With stratified sampling
# train_image, test_image, train_label, test_label = read_data('zh-plus/tiny-imagenet', sample=True, ratio=0.3)

# Create an instance of the ResNet model with 3 residual blocks
model = Model(
    in_channels=train_image.size(1),
    out_channels=[64, 64, 128, 256],
    number_of_blocks=3,
    number_of_classes=torch.unique(train_label).size(0), 
    number_of_epochs=20,
    batch_size=128,
    learning_rate=2e-3,
    weight_decay=2e-4
)

# Train and evaluate the model
start = time.time()
print('Training and evaluation in progress...')
model.fit(train_image, test_image, train_label, test_label)
print(f'Model has finished training and evaluation. Runtime: {(time.time() - start) / 60:.4f} minutes')

# Plot both training and testing learning curves
plt.plot(model.train_accurcay, color='blue', marker='o', label='Train ACC')
plt.plot(model.test_accurcay, color='red', marker='o', label='Test ACC')
plt.axvline(x=np.argmax(model.test_accurcay), linestyle='--', color='black', label='Best Test ACC')

plt.title('Learning curves')
plt.xticks(range(0, 22, 2), range(0, 22, 2))
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig('learning_curve.jpg', dpi=300, bbox_inches='tight')
plt.close()