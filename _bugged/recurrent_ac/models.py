import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

FC1_DIMS = 1024
FC2_DIMS = 512
HIDDEN_STATE_SIZE = 32

class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.data_shape = 4
        self.action_shape = 2

        self.fc1_dims = FC1_DIMS
        self.fc2_dims = FC2_DIMS
        self.hidden_state_size = HIDDEN_STATE_SIZE

        self.input_shape = self.data_shape + self.hidden_state_size
        self.output_shape = self.action_shape + self.hidden_state_size


        self.net = nn.Sequential(
            nn.Linear(self.input_shape, self.fc1_dims),    nn.ReLU(), 
            nn.Linear(self.fc1_dims, self.fc2_dims),       nn.ReLU(),
            nn.Linear(self.fc2_dims, self.output_shape)
        )

    def get_new_hidden_state(self):
        return torch.zeros(self.hidden_state_size)

    def forward(self, state, hidden_state):
        y = self.net(torch.cat([state, hidden_state]))
        actions = y[:self.action_shape]
        hidden_state = y[self.action_shape:]

        return actions, hidden_state

class Critic(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.data_shape = 4
        self.action_shape = 1

        self.fc1_dims = FC1_DIMS
        self.fc2_dims = FC2_DIMS
        self.hidden_state_size = HIDDEN_STATE_SIZE

        self.input_shape = sum([
            self.data_shape,
            self.action_shape,
            self.hidden_state_size
        ])
        self.output_shape = self.hidden_state_size + 1


        self.net = nn.Sequential(
            nn.Linear(self.input_shape, self.fc1_dims),    nn.ReLU(), 
            nn.Linear(self.fc1_dims, self.fc2_dims),       nn.ReLU(),
            nn.Linear(self.fc2_dims, self.output_shape)
        )

    def get_new_hidden_state(self):
        return torch.zeros(self.hidden_state_size)

    def forward(self, state, action, hidden_state):
        y = self.net(torch.cat([state, action, hidden_state]))
        value, hidden_state = y[0], y[1:]

        return value, hidden_state