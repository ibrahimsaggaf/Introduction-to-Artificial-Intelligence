import csv
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def scale(X):

    # Scale the features to the range from 0 and 1
    scaler = MinMaxScaler()

    return scaler.fit_transform(X)


def split(data, size=0.8):

    # Time-based split
    idx = round(data.shape[0] * size)
    train = data[:idx, :]
    test = data[idx:, :]

    return train, test


def read_data(data_name):
    et = []
    with open(data_name, 'r') as f:
        reader = csv.reader(f)

        # Remove header
        _ = reader.__next__()

        for row in reader:
            et.append(row[1:]) # Remove the date column

    et = np.array(et, dtype=np.float32)
    et = scale(et)
    train, test = split(et)
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]

    return train_X, test_X, train_y, test_y


def data_loader(X, y, lag):
    X = torch.tensor(X, dtype=torch.float32).view(X.shape[0], 1, X.shape[1])
    y = torch.tensor(y, dtype=torch.float32).view(y.shape[0], 1, 1)

    for i in range(X.size(0) - lag):
        yield X[i: i + lag, :], y[i + lag]