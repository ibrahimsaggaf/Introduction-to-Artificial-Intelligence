'''
Author: Dr Ibrahim Alsaggaf
Learning type: Supervised learning
Task: Binary classification
Dataset: Breast Cancer Wisconsin (Diagnostic) [1]
Library: Scikit-learn [2]
Model: Random Forest

[1] Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic) [Dataset]. 
    UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B.

[2] Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
'''


from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

from utils import scale, split, performance

# Load the dataset from the Scikit-learn library
dataset = load_breast_cancer()
X, y = dataset.data, dataset.target

# Scale the features by removing the mean and unit variance (mean=0, variance=1)
X = scale(X)

# Split the dataset into two sets: A training set and a test set, where the first is used to train
# The model whilst the second one is used to evaluate the model's performance
train_X, test_X, train_y, test_y = split(X, y)

# Create an instance of the model
model = RandomForestClassifier(random_state=1111)

# Train the model
model.fit(train_X, train_y)

# Obtain the model's prediction
prediction = model.predict(test_X)

# Evaluate the model's performance
f1_score = performance(test_y, prediction, type='supervised')
print(f'F1 score: {f1_score:.4f}')