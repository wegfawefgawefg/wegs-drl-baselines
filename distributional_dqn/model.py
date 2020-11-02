import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Network(torch.nn.Module):
    def __init__(self, alpha, input_shape, num_actions,
            support, num_atoms):
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions

        self.support = support
        self.num_atoms = num_atoms

        self.fc1_dims = 1024
        self.fc2_dims = 512

        self.layers = nn.Sequential(
            nn.Linear(*self.input_shape, self.fc1_dims),
            nn.ReLU(),
            nn.Linear( self.fc1_dims,    self.fc2_dims),
            nn.ReLU(),
            nn.Linear( self.fc2_dims,    self.num_actions * self.num_atoms),
            nn.ReLU(),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def dist(self, x):
        q_atoms = self.layers(x)
        q_atoms = q_atoms.view(-1, self.num_actions, self.num_atoms)
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)

        return dist
        
    def forward(self, x):
        dist = self.dist(x)
        scaled_dist = dist * self.support
        qs = torch.sum(scaled_dist, dim=2)

        return qs

if __name__ == "__main__":
    V_MIN, V_MAX = 0.0, 200.0
    NUM_ATOMS = 8
    linspace = torch.linspace(V_MIN, V_MAX, NUM_ATOMS)

    net = Network(
        alpha=0.01,
        input_shape=(4,),
        num_actions=2,
        support=linspace,
        num_atoms=NUM_ATOMS)    

    state = torch.ones((1,4))
    pred = net(state)
    print(pred)