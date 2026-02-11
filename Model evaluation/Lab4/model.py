import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader, TensorDataset

from network import MLP


class Model:
    def __init__(self, 
                 input_dimension, 
                 hidden_dim, 
                 output_dimension, 
                 number_of_hidden,
                 batch_size,
                 number_of_epochs,
                 size
        ):
        self.input_dimension = input_dimension
        self.hidden_dim = hidden_dim
        self.output_dimension = output_dimension
        self.number_of_hidden = number_of_hidden
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs
        self.size = size

        self.train_loss = []
        self.test_loss = []


    def __weights_init(self, net):
        if type(net) == nn.Linear:
            net.weight.data.normal_(0.0, 0.02)
            net.bias.data.fill_(0)
        

    def __get_new_model(self):
        self.mlp = MLP(
            self.input_dimension,
            self.hidden_dim,
            self.output_dimension,
            self.number_of_hidden,
        )
        self.mlp.apply(self.__weights_init)
        self.opt = opt.Adam(self.mlp.parameters())
        self.criterion = nn.CrossEntropyLoss()
    
    
    def __train(self, train_loader):

        # Measure training loss
        epoch_loss, count = 0, 0
        self.mlp.train()

        for train_X, train_y in train_loader:
            self.opt.zero_grad()
            prediction = self.mlp(train_X)
            loss = self.criterion(prediction, train_y)
            loss.backward()
            self.opt.step()

            epoch_loss += loss.item()
            count += 1

        self.train_loss.append(epoch_loss / count)


    def __evaluate(self, test_loader):
        
        # Measure testing loss
        epoch_loss, count = 0, 0
        self.mlp.eval()
        
        with torch.no_grad():
            for test_X, test_y in test_loader:
                prediction = self.mlp(test_X)
                loss = self.criterion(prediction, test_y)

                epoch_loss += loss.item()
                count += 1

        self.test_loss.append(epoch_loss / count)
    
    
    def fit(self, train_X, test_X, train_y, test_y):
        train_data = TensorDataset(train_X, train_y)
        train_loader = DataLoader(train_data, batch_size=self.batch_size)
        test_data = TensorDataset(test_X, test_y)
        test_loader = DataLoader(test_data, batch_size=self.batch_size)
        self.__get_new_model()

        for epoch in range(self.number_of_epochs + 1):
            self.__train(train_loader)
            self.__evaluate(test_loader)
            print(f'{self.size}>> Epoch: {epoch}, Train loss: {self.train_loss[-1]:.4f}, Test loss: {self.test_loss[-1]:.4f}')

