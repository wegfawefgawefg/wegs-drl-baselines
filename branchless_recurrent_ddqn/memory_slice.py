import math
import random

import gym       
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MemorySlice:
    def __init__(self, slice_size, state_shape, action_shape):
        self.size = slice_size
        self.count = 0

        self.states       = np.zeros((self.size,  *state_shape),  dtype=np.float32)
        self.actions      = np.zeros((self.size, *action_shape),  dtype=np.float32)
        self.rewards      = np.zeros((self.size,             1),  dtype=np.float32)
        self.next_states  = np.zeros((self.size,  *state_shape),  dtype=np.float32)
        self.dones        = np.zeros((self.size,             1),  dtype=np.bool   )

    def store_transition(self, state, action, reward, next_state, done):
        if self.count == self.size:
            raise Exception("slice already full. Save it to your slice replay buffer and make a new one")

        index = self.count
        self.states[index]      = state
        self.actions[index]     = action
        self.rewards[index]     = reward
        self.next_states[index] = next_state
        self.dones[index]       = done

        self.count += 1

    def get(self):
        if self.count < self.size - 1:
            print(self.state_memory)
            print(self.state_memory.shape)
            raise Exception(
                "attempting to get() incomplete slice. please fill it before calling get()")
        return self.states, self.actions, self.rewards, self.next_states, self.dones