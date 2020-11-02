from model import Network
from np_replay_buffer import PrioritizedReplayBuffer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from cpprb import PrioritizedReplayBuffer

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
        self.importance_exp = Lerper(start=0.4, end=1.0, num_steps=100000)

        self.priority_exp = 0.6
        self.memory = PrioritizedReplayBuffer(
            max_mem_size, 
            {   "obs":      { "shape": state_shape  },
                "act":      { "shape": 1            },
                "rew":      {                       },
                "next_obs": { "shape": state_shape  },
                "done":     { "shape": 1            }
            },
            alpha = self.priority_exp)

        self.net = Network(lr, state_shape, num_actions)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon.value():
            state = torch.tensor(observation).float().detach()
            state = state.to(self.net.device)
            state = state.unsqueeze(0)

            q_values = self.net(state)
            action = torch.argmax(q_values).item()
            return action
        else:
            return np.random.choice(self.action_space)

    def store_memory(self, state, action, reward, next_state, done):
        self.memory.add(obs=state, act=action, rew=reward, next_obs=next_state, done=done)  

    def learn(self):
        if self.memory.get_stored_size() < self.batch_size:
            return
    
        batch = self.memory.sample(self.batch_size, self.importance_exp.value())
            
        states  = torch.tensor( batch["obs"]                     ).to(self.net.device)
        actions = torch.tensor( batch["act"],   dtype=torch.int64).to(self.net.device).T[0]
        rewards = torch.tensor( batch["rew"]                     ).to(self.net.device).T[0]
        states_ = torch.tensor( batch["next_obs"]                ).to(self.net.device)
        dones   = torch.tensor( batch["done"],  dtype=torch.bool ).to(self.net.device).T[0]
        weights = torch.tensor( batch["weights"]                 ).to(self.net.device)

        batch_index = np.arange(self.batch_size, dtype=np.int64)

        q_values  =   self.net(states)[batch_index, actions]
        q_values_ =   self.net(states_)

        action_qs_ = torch.max(q_values_, dim=1)[0]
        action_qs_[dones] = 0.0
        q_target = rewards + self.gamma * action_qs_

        td = q_target - q_values

        self.net.optimizer.zero_grad()
        loss = ((td ** 2.0) * weights).mean()
        loss.backward()
        self.net.optimizer.step()

        new_priorities = (td.abs()).detach().cpu()
        self.memory.update_priorities(batch["indexes"], new_priorities)

        self.epsilon.step()
        self.importance_exp.step()