import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader, TensorDataset

from network import ResNet


class Model:
    def __init__(self,
                 in_channels,
                 out_channels,
                 number_of_blocks,
                 number_of_classes, 
                 number_of_epochs,
                 batch_size,
                 learning_rate,
                 weight_decay
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.number_of_blocks = number_of_blocks
        self.number_of_classes = number_of_classes
        self.number_of_epochs = number_of_epochs
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.train_accurcay = []
        self.test_accurcay = []


    def __weights_init(self, net):
        if type(net) == nn.Linear:
            net.weight.data.normal_(0.0, 0.01)


    def __get_new_model(self):
        self.resnet = ResNet(
            self.in_channels,
            self.out_channels,
            self.number_of_blocks,
            self.number_of_classes
        )
        self.resnet.apply(self.__weights_init)
        self.opt = opt.Adam(
            self.resnet.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()


    def __train(self, train_loader):

        # Measure training accuracy
        correct, count = 0, 0
        self.resnet.train()

        for train_image, train_label in train_loader:
            self.opt.zero_grad()
            out = self.resnet(train_image)
            loss = self.criterion(out, train_label)
            loss.backward()
            self.opt.step()

            _, prediction = out.max(dim=1)
            count += train_label.size(0)
            correct += prediction.eq(train_label).sum().item()

        self.train_accurcay.append((correct / count) * 100.0)


    def __evaluate(self, test_loader):
        
        # Measure testing accuracy
        correct, count = 0, 0
        self.resnet.eval()
        
        with torch.no_grad():
            for test_image, test_label in test_loader:
                out = self.resnet(test_image)
                _, prediction = out.max(dim=1)
                count += test_label.size(0)
                correct += prediction.eq(test_label).sum().item()

        self.test_accurcay.append((correct / count) * 100.0)
    

    def fit(self, train_image, test_image, train_label, test_label):
        train_data = TensorDataset(train_image, train_label)
        train_loader = DataLoader(train_data, batch_size=self.batch_size)
        test_data = TensorDataset(test_image, test_label)
        test_loader = DataLoader(test_data, batch_size=self.batch_size)
        self.__get_new_model()

        # Check model size (number of parameters)
        model_size = sum(p.numel() for p in self.resnet.parameters())
        print(f'Model size: {round(model_size / 1e6, 2)}M parameters')

        for epoch in range(self.number_of_epochs + 1):
            self.__train(train_loader)
            self.__evaluate(test_loader)
            print(
                f'Epoch: {epoch}, Train ACC: {self.train_accurcay[-1]:.4f}' \
                f' , Test ACC: {self.test_accurcay[-1]:.4f}'
            )