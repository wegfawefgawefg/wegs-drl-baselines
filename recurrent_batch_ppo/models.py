import math
import random

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

FC1_DIMS = 512
FC2_DIMS = 256

class LSTM(torch.nn.Module):
    def __init__(self, state_shape, hidden_state_size):
        super().__init__()

        self.data_shape = state_shape[0]
        self.hidden_state_size = hidden_state_size

        self.lstm = nn.LSTM(input_size=self.data_shape, hidden_size=self.hidden_state_size,
            num_layers=1, bias=True, batch_first=True)

    def forward(self, state, hidden_state):
        outputs, hiddens = self.lstm(state, hidden_state)
        return outputs, hiddens

class Actor(torch.nn.Module):
    def __init__(self, state_shape, action_shape):
        super(Actor, self).__init__()

        self.data_shape = state_shape
        self.action_shape = action_shape
        
        self.std = 0.01

        self.fc1_dims = FC1_DIMS
        self.fc2_dims = FC2_DIMS
        
        self.net = nn.Sequential(
            nn.Linear(   self.data_shape,    self.fc1_dims      ),  nn.ReLU(), 
            nn.Linear(   self.fc1_dims,      self.fc2_dims      ),  nn.ReLU(),
            nn.Linear(   self.fc2_dims,     *self.action_shape  ),
        )
        
        self.log_std = nn.Parameter(torch.ones(1, *self.action_shape) * self.std)

    def forward(self, x):
        mu    = self.net(x)
        std   = self.log_std.exp().expand_as(mu)
        policy_dist  = torch.distributions.Normal(mu, std)
        return policy_dist
    
    def forward_deterministic(self, x):
        mu = self.net(x)
        return mu

class Critic(nn.Module):
    def __init__(self, state_shape):
        super(Critic, self).__init__()
        
        self.data_shape = state_shape

        self.fc1_dims = FC1_DIMS
        self.fc2_dims = FC2_DIMS
        
        self.net = nn.Sequential(
            nn.Linear(   self.data_shape,   self.fc1_dims   ),  nn.ReLU(), 
            nn.Linear(   self.fc1_dims,     self.fc2_dims   ),  nn.ReLU(),
            nn.Linear(   self.fc2_dims,     1               ),         
        )

    def forward(self, x):
        value = self.net(x)
        return value

if __name__ == "__main__": 
    BATCH_SIZE = 2
    RECURRENCE_N_STEPS = 4
    STATE_SHAPE = (8,)
    HIDDEN_SIZE = 256

    lstm = LSTM(state_shape=STATE_SHAPE, hidden_state_size=HIDDEN_SIZE)

    fake_data = torch.zeros((BATCH_SIZE, RECURRENCE_N_STEPS, *STATE_SHAPE))
    fake_data[0, :] += 1.0
    hidden_state    = torch.zeros((1, BATCH_SIZE, HIDDEN_SIZE))
    cell_state      = torch.zeros((1, BATCH_SIZE, HIDDEN_SIZE))

    output, (batch_hidden, batch_cell) = lstm(fake_data, (hidden_state, cell_state))

    from pprint import pprint
    pprint(output)
    print(output.shape)
    print(batch_hidden.shape)
    print(batch_cell.shape)