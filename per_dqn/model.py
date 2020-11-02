import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

class Network(torch.nn.Module):
    def __init__(self, alpha, inputShape, numActions):
        super().__init__()
        self.inputShape = inputShape
        self.numActions = numActions
        self.fc1Dims = 1024
        self.fc2Dims = 512

        # print("input shape {}".format(self.inputShape))
        self.fc1 = nn.Linear(*self.inputShape, self.fc1Dims)
        self.fc2 = nn.Linear(self.fc1Dims, self.fc2Dims)
        self.fc3 = nn.Linear(self.fc2Dims, numActions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

if __name__ == "__main__":
    numStackedFrames = 4
    inputShape = (1, 8)

    net = Network(
        alpha=0.001, 
        inputShape=8, 
        numActions=3)

    #   make fake data
    x = torch.ones(inputShape).to(net.device)
    print(x.shape)

    #   feedforward 
    qValues = net(x)
    print("qValues {}".format(qValues))
