import torch
import numpy as np


def read_data(data_name):
    mnist = np.loadtxt(data_name, delimiter=',')
    X = mnist[:, 1:]
    y = mnist[:, 0]

    # Scale images by dividing by 255
    X /= 255.0

    return torch.tensor(X, dtype=torch.float32), torch.LongTensor(y)
