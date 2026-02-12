import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, 
                 input_size, 
                 output_size,
                 hidden_size, 
                 num_layers, 
                 dropout
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size * num_layers, output_size)
        self.relu = nn.ReLU()


    def forward(self, v_t):
        h_t = torch.zeros(self.num_layers, v_t.size(0), self.hidden_size)
        c_t = torch.zeros(self.num_layers, v_t.size(0), self.hidden_size)

        _, (h_t2, c_t2) = self.lstm(v_t, (h_t, c_t))
        h_t2 = h_t2.transpose(0, 1)
        h_t2 = h_t2.reshape(-1, self.hidden_size * self.num_layers)
        h_t2 = self.relu(h_t2)
        v_t2 = self.linear(h_t2)

        return v_t2