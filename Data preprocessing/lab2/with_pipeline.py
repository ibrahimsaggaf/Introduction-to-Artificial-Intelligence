'''
Author: Dr Ibrahim Alsaggaf
Learning type: Supervised learning
Task: Regression with data preprocessing
Dataset: Diabetes dataset [1]
Library: Scikit-learn [2]
Model: Support Vector Machine (SVM)

[1] Efron, B., Hastie, T., Johnstone, I., & Tibshirani, R. (2004). Least angle regression. 
    https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html

[2] Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
'''


from sklearn.svm import SVR
from sklearn.datasets import load_diabetes

from utils import split, scale, feature_selection, performance

# Load the dataset from the Scikit-learn library
dataset = load_diabetes()
X, y = dataset.data, dataset.target

# Scale the features by removing the mean and unit variance (mean=0, variance=1)
X = scale(X)

# Select the top 2 important features using a tree-based model
print(f'The number of features before selection is {X.shape[1]}')
X = feature_selection(X, y, top=2)
print(f'The number of features after selection is {X.shape[1]}')

# Split the dataset into two sets: A training set and a test set, where the first is used to train
# The model whilst the second one is used to evaluate the model's performance
train_X, test_X, train_y, test_y = split(X, y)

# Create an instance of the model
model = SVR(kernel='linear')

# Train the model
model.fit(train_X, train_y)

# Obtain the model's prediction
prediction = model.predict(test_X)

# Evaluate the model's performance
mse = performance(test_y, prediction)
print(f'MSE: {mse:.4f}')