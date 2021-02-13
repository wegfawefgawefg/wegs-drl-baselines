import math
import random

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

FC1_DIMS = 256
FC2_DIMS = 128

class Actor(torch.nn.Module):
    def __init__(self, state_shape, action_shape):
        super(Actor, self).__init__()

        self.data_shape = state_shape
        self.action_shape = action_shape
        
        self.std = 0.0

        self.fc1_dims = FC1_DIMS
        self.fc2_dims = FC2_DIMS
        
        self.net = nn.Sequential(
            nn.Linear(  *self.data_shape,    self.fc1_dims      ),  nn.ReLU(), 
            nn.Linear(   self.fc1_dims,      self.fc2_dims      ),  nn.ReLU(),
            nn.Linear(   self.fc2_dims,     *self.action_shape  ),
        )
        
        self.log_std = nn.Parameter(torch.ones(1, *self.action_shape) * self.std)

    def forward(self, x):
        mu    = self.net(x)
        std   = self.log_std.exp().expand_as(mu)
        policy_dist  = torch.distributions.Normal(mu, std)
        return policy_dist

class Critic(nn.Module):
    def __init__(self, state_shape):
        super(Critic, self).__init__()
        
        self.data_shape = state_shape

        self.fc1_dims = FC1_DIMS
        self.fc2_dims = FC2_DIMS
        
        self.net = nn.Sequential(
            nn.Linear(  *self.data_shape,   self.fc1_dims   ),  nn.ReLU(), 
            nn.Linear(   self.fc1_dims,     self.fc2_dims   ),  nn.ReLU(),
            nn.Linear(   self.fc2_dims,     1               ),         
        )

    def forward(self, x):
        value = self.net(x)
        return value