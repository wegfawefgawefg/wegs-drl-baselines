import math
import random

import gym       
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Trajectory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def store_transition(self, state, action, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()

    def fetch_on_device(self, device):
        states  = torch.tensor(np.stack(self.states), dtype=torch.float32).to(device)
        actions = torch.tensor(np.stack(self.actions), dtype=torch.int64).to(device)
        rewards = torch.tensor(np.stack(self.rewards), dtype=torch.float32).to(device)
        dones   = torch.tensor(np.stack(self.dones), dtype=torch.bool).to(device)

        return states, actions, rewards, dones