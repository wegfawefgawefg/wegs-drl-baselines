from model import Network
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from cpprb import ReplayBuffer

class Lerper:
    def __init__(self, start, end, num_steps):
        self.delta = (end - start) / float(num_steps)
        self.num = start - self.delta
        self.count = 0
        self.num_steps = num_steps

    def value(self):
        return self.num

    def step(self):
        if self.count <= self.num_steps:
            self.num += self.delta
        self.count += 1
        return self.num

class Agent:
    def __init__(self, lr, state_shape, num_actions, batch_size, 
            max_mem_size=100000):
        self.lr = lr
        self.gamma = 0.99
        self.action_space = list(range(num_actions))
        self.batch_size = batch_size

        self.epsilon = Lerper(start=1.0, end=0.01, num_steps=2000)

        self.memory = ReplayBuffer(
            max_mem_size, 
            {   "obs":      { "shape": state_shape  },
                "act":      { "shape": 1            },
                "rew":      {                       },
                "next_obs": { "shape": state_shape  },
                "done":     { "shape": 1            }})

        self.net = Network(lr, state_shape, num_actions)

    def choose_action(self, observation):
        state = torch.tensor(observation).float().detach()
        state = state.to(self.net.device)
        state = state.unsqueeze(0)

        q_values = self.net(state)
        action = torch.argmax(q_values).item()
        return action

    def store_memory(self, state, action, reward, next_state, done):
        self.memory.add(obs=state, act=action, rew=reward, next_obs=next_state, done=done)  

    def learn(self):
        if self.memory.get_stored_size() < self.batch_size:
            return
    
        batch = self.memory.sample(self.batch_size)
            
        states  = torch.tensor( batch["obs"]                     ).to(self.net.device)
        actions = torch.tensor( batch["act"],   dtype=torch.int64).to(self.net.device).T[0]
        rewards = torch.tensor( batch["rew"]                     ).to(self.net.device).T[0]
        states_ = torch.tensor( batch["next_obs"]                ).to(self.net.device)
        dones   = torch.tensor( batch["done"],  dtype=torch.bool ).to(self.net.device).T[0]

        batch_index = np.arange(self.batch_size, dtype=np.int64)

        q_values  =   self.net(states)[batch_index, actions]
        q_values_ =   self.net(states_)

        action_qs_ = torch.max(q_values_, dim=1)[0]
        action_qs_[dones] = 0.0
        q_target = rewards + self.gamma * action_qs_

        td = q_target - q_values

        self.net.optimizer.zero_grad()
        loss = (td ** 2.0).mean()
        loss.backward()
        self.net.optimizer.step()

        self.net.reset_noise()