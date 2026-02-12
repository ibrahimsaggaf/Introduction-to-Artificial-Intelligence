import time
import torch
import torch.nn as nn
import torch.optim as opt

from utils import data_loader
from network import LSTM


class Model:
    def __init__(self,
                 lag,
                 input_size,
                 output_size, 
                 hidden_size, 
                 number_of_layers, 
                 dropout, 
                 number_of_epochs,
                 learning_rate
    ):
        self.lag = lag
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.number_of_layers = number_of_layers
        self.dropout = dropout
        self.number_of_epochs = number_of_epochs
        self.learning_rate = learning_rate


    def __weights_init(self, net):
        if type(net) == nn.Linear:
            net.weight.data.normal_(0.0, 0.01)


    def __get_new_model(self):
        self.lstm = LSTM(
            self.input_size,
            self.output_size,
            self.hidden_size, 
            self.number_of_layers, 
            self.dropout
        )
        self.lstm.apply(self.__weights_init)
        self.opt = opt.Adam(self.lstm.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()


    def __train(self, train_loader):

        # Training the model
        self.lstm.train()

        for train_X, train_y in train_loader:
            self.opt.zero_grad()
            prediction = self.lstm(train_X)
            loss = self.criterion(prediction, train_y)
            loss.backward()
            self.opt.step()


    def fit(self, train_X, train_y):
        train_loader = data_loader(train_X, train_y, self.lag)
        self.__get_new_model()

        for epoch in range(self.number_of_epochs + 1):
            self.__train(train_loader)


    def predict(self, X, y):
        preds = []
        data_loader_ = data_loader(X, y, self.lag)
        self.lstm.eval()
        
        with torch.no_grad():
            for X, _ in data_loader_:
                prediction = self.lstm(X)
                preds.append(prediction.item())

        return preds