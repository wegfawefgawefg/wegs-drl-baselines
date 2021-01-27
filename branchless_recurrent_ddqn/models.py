
import math
import random

import gym       
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


FC1_DIMS = 1024
FC2_DIMS = 512

HIDDEN_STATE_SIZE = 512

class DQN(torch.nn.Module):
    def __init__(self, state_shape, num_actions):
        super().__init__()

        self.data_shape = state_shape[0]
        self.num_actions = num_actions

        self.fc1_dims = FC1_DIMS
        self.fc2_dims = FC2_DIMS
        self.hidden_state_size = HIDDEN_STATE_SIZE

        self.rnn = nn.RNN(input_size=self.data_shape, hidden_size=self.hidden_state_size,
            num_layers=1, nonlinearity="relu", bias=True, batch_first=True)

        self.net = nn.Sequential(
            nn.Linear(self.hidden_state_size, self.fc1_dims),   nn.ReLU(), 
            nn.Linear(self.fc1_dims, self.fc2_dims),            nn.ReLU(),
            nn.Linear(self.fc2_dims, self.num_actions)
        )

    def forward(self, state, hidden_state):
        _, last_hiddens = self.rnn(state, hidden_state)
        qs = self.net(last_hiddens)[0]
        return qs, last_hiddens

    def get_batch_hidden_state(self, batch_size):
        return torch.zeros((1, batch_size, self.hidden_state_size))

if __name__ == "__main__":

    BATCH_SIZE = 2
    SLICE_SIZE = 6
    STATE_SHAPE = (4,)

    net = DQN(STATE_SHAPE, num_actions=2)

    fake_data = torch.zeros((BATCH_SIZE, SLICE_SIZE, *STATE_SHAPE))
    fake_data[0, :] += 1.0
    hidden_state = torch.zeros((1, BATCH_SIZE, net.hidden_state_size))

    qs, _ = net(fake_data, hidden_state)

    print(qs)
    print(qs.shape)