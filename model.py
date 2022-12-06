import torch.nn as nn
import torch.nn.functional as F


class TwoLayerModel(nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        # Create two layer network, where each layer has same number of neurons as LogisticRegression
        # model used in Baseline.
        # Since LogReg uses input_dims neurons, we can make hidden_dims = input_dims
        ##############################################################################################
        # CONFIGURATIONS
        hidden_dims = input_dims
        self.relu1 = nn.Sigmoid()# nn.ReLU()
        self.relu2 = nn.Tanh()
        
        ##############################################################################################
        
        self.fc1 = nn.Linear(input_dims, hidden_dims, bias=True)
        nn.init.xavier_uniform(self.fc1.weight)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims, bias=True)
        nn.init.xavier_uniform(self.fc2.weight)
        self.output_linear= nn.Linear(hidden_dims, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.output_linear(x)
        x = self.sigmoid(x)
        # put sigmoid?
        return x


class TestModel(nn.Module):
    def __init__(self, input_size):
        super(TestModel, self).__init__()
        # Number of input features is 215.
        self.layer_1 = nn.Linear(input_size, 256)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity='relu')
        self.layer_2 = nn.Linear(256, 128)
        nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity='relu')
        self.layer_3 = nn.Linear(128, 64)
        nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity='relu')
        self.layer_out = nn.Linear(64, 1)
        nn.init.xavier_uniform_(self.layer_out.weight)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(256)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)
        
        self.sigmoid = nn.Sigmoid()
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.relu(self.layer_3(x))
        x = self.batchnorm3(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        x = self.sigmoid(x)
        return x