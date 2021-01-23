import math
import random

import gym       
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.policies = []
        self.rewards = []
        self.dones = []
        self.actor_hidden_states = []
        self.action_log_probs = []

    def store(self, state, action, policy, 
            reward, done, actor_hidden_state, action_log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.policies.append(policy)
        self.rewards.append(reward)
        self.dones.append(done)
        self.actor_hidden_states.append(actor_hidden_state)
        self.action_log_probs.append(action_log_prob)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.policies.clear()
        self.rewards.clear()
        self.dones.clear()
        self.actor_hidden_states.clear()
        self.action_log_probs.clear()

    def fetch_on_device(self, device):
        states  = torch.tensor(np.stack(self.states), dtype=torch.float32).to(device)
        actions = torch.tensor(np.stack(self.actions), dtype=torch.float32).to(device)
        rewards = torch.tensor(np.stack(self.rewards), dtype=torch.float32).to(device)
        dones   = torch.tensor(np.stack(self.dones), dtype=torch.float32).to(device)
        
        policies            = torch.stack(self.policies).to(device).to(device)
        actor_hidden_states = torch.stack(self.actor_hidden_states).to(device)
        action_log_probs    = torch.stack(self.action_log_probs).to(device)

        return states, actions, policies, rewards, dones, actor_hidden_states, action_log_probs