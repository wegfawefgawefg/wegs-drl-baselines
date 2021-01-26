import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# FC1_DIMS = 1024
# FC2_DIMS = 512
# HIDDEN_STATE_SIZE = 32

FC1_DIMS = 512
FC2_DIMS = 256
HIDDEN_STATE_SIZE = 32

class DQN(torch.nn.Module):
    def __init__(self, state_shape, num_actions):
        super().__init__()

        self.data_shape = state_shape[0]
        self.num_actions = num_actions

        self.fc1_dims = FC1_DIMS
        self.fc2_dims = FC2_DIMS
        self.hidden_state_size = HIDDEN_STATE_SIZE

        self.input_shape = self.data_shape + self.hidden_state_size
        self.output_shape = self.num_actions + self.hidden_state_size


        self.net = nn.Sequential(
            nn.Linear(self.input_shape, self.fc1_dims),    nn.ReLU(), 
            nn.Linear(self.fc1_dims, self.fc2_dims),       nn.ReLU(),
            nn.Linear(self.fc2_dims, self.output_shape)
        )

    def get_new_hidden_state(self):
        return torch.zeros((1, self.hidden_state_size))

    def get_batch_hidden_state(self, batch_size):
        return torch.zeros((batch_size, self.hidden_state_size))

    def forward(self, state, hidden_state):
        y = self.net(torch.cat([state, hidden_state], dim=1))
        actions = y[:, :self.num_actions]
        hidden_state = y[:, self.num_actions:]

        return actions, hidden_state