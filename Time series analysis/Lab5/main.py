'''
Author: Dr Ibrahim Alsaggaf
Learning type: Supervised learning
Task: Forecasting multivariate time series
Dataset: The Electricity Transformer Dataset [1]
Libraries:
    Scikit-learn [2]
    Pytorch [3]
    Matplotlib [4]

Model: LSTM

[1] Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021). 
    Informer: Beyond efficient transformer for long sequence time-series forecasting. 
    In Proceedings of the AAAI conference on artificial intelligence (Vol. 35, No. 12, 
    pp. 11106-11115).
[2] Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
[3] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). 
    Pytorch: An imperative style, high-performance deep learning library. Advances in neural information 
    processing systems, 32.
[4] Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in science & engineering, 9(03), 90-95.
'''

import time
import numpy as np
from matplotlib import pyplot as plt

from model import Model
from utils import read_data

# Load the ET dataset
start = time.time()
train_X, test_X, train_y, test_y = read_data('ETTm1.csv')
print(f'Data is loaded. Runtime: {(time.time() - start) / 60:.4f} minutes')

# Create an instance of the model with only 2 LSTM layers
model = Model(
    lag=1,
    input_size=train_X.shape[1],
    output_size=1,
    hidden_size=4, 
    number_of_layers=4, 
    dropout=0.0,
    number_of_epochs=1000,
    learning_rate=0.002
)

# Train the model
start = time.time()
print('Training in progress...')
model.fit(train_X, train_y)
print(f'Model has finished training. Runtime: {(time.time() - start) / 60:.4f} minutes')

# Predict the series
start = time.time()
train_preds = model.predict(train_X, train_y)
test_preds = model.predict(test_X, test_y)
print(f'Model has finished prediction. Runtime: {(time.time() - start) / 60:.4f} minutes')

# Plot both real and predicted target series (OT)
plt.figure(figsize=(20, 8))
plt.plot(np.hstack((train_y, test_y)), color='blue', label='OT (Traget)')
plt.plot(train_preds, color='green', label='Train prediction')
plt.plot(
    range(len(train_preds), len(train_preds) + len(test_preds)), 
    test_preds, color='red', label='Test prediction'
)

plt.title('Oil Temperature series')
plt.xticks([])
plt.xlabel('Time')
plt.ylabel('Scaled temperature')
plt.legend()

plt.savefig('forecasting.jpg', dpi=300, bbox_inches='tight')
plt.close()