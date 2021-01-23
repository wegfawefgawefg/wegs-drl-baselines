import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Rec(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.data_shape = 5
        self.hidden_state_size = 64
        self.input_shape = self.data_shape + self.hidden_state_size
        self.output_shape = self.hidden_state_size + 1

        self.hidden_state = None
        self.reset_hidden_state()

        self.fc1_dims = 64
        self.fc2_dims = 64

        self.net = nn.Sequential(
            nn.Linear(self.input_shape, self.fc1_dims),    nn.ReLU(), 
            nn.Linear(self.fc1_dims, self.fc2_dims),       nn.ReLU(),
            nn.Linear(self.fc2_dims, self.output_shape)
        )

    def reset_hidden_state(self):
        self.hidden_state = torch.zeros(self.hidden_state_size)

    def forward(self, x):
        num_data = x.shape[0]

        self.reset_hidden_state()

        for i in range(num_data):
            whole_input = torch.cat([x[i], self.hidden_state])
            y = self.net(whole_input)
            action, self.hidden_state = y[0], y[1:]

        return action

if __name__ == "__main__":
    fake_data = torch.rand((64, 5))
    net = Rec()
    action = net(fake_data)

    print(action)

    action.backward()
    