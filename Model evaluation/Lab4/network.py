import torch.nn as nn


class LinearLayer(nn.Module):
    def __init__(self, 
                 input_dimension, 
                 output_dimension, 
                 batch_norm=True, 
                 activation=True
        ):

        super().__init__()

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.batch_norm = batch_norm
        self.activation = activation

        self.linear = nn.Linear(self.input_dimension, self.output_dimension)
        if self.batch_norm:
            self.norm = nn.BatchNorm1d(self.output_dimension)
        
        if self.activation:
            self.relu = nn.ReLU()


    def forward(self, x):
        x = self.linear(x)
        if self.batch_norm:
            x = self.norm(x)

        if self.activation:
            x = self.relu(x)

        return x


class MLP(nn.Module):
    def __init__(self, 
                 input_dimension, 
                 hidden_dim, 
                 output_dimension, 
                 number_of_hidden
        ):

        super().__init__()

        layers = []
        for i in range(number_of_hidden):
            in_ = input_dimension if i == 0 else hidden_dim
            out_ = output_dimension if i == number_of_hidden -1 else hidden_dim

            if i ==  number_of_hidden - 1:
                layers += [LinearLayer(in_, out_, batch_norm=False, activation=False)]

            else:
                layers += [LinearLayer(in_, out_)]

        self.mlp = nn.Sequential(*layers)


    def forward(self, x):

        return self.mlp(x)