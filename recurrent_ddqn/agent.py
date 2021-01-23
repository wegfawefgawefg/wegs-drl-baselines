import math
import random
import copy

from pprint import pprint
import gym       
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models import DQN
from trajectory import Trajectory

class LinearSchedule:
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

class Agent():
    def __init__(self, learn_rate, input_shape, num_actions, 
            batch_size=32):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.gamma = 0.99
        self.tau = 1.0

        self.memories = []
        self.epsilon = LinearSchedule(start=1.0, end=0.01, num_steps=2000)

        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

        self.net = DQN().to(self.device)
        self.target_net  = copy.deepcopy(self.net).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learn_rate)

        self.loss = nn.MSELoss()

    def update_target_net_params(self):
        for param, target_param in zip(self.net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def choose_action(self, observation, hidden_state):
            state = torch.tensor(observation).float().detach()
            state = state.to(self.device)

            q_values, hidden_state_ = self.net(state, hidden_state)
            action = torch.argmax(q_values).item()

            if random.random() <= self.epsilon.value():
                action = random.randint(0, self.num_actions - 1)

            return action, hidden_state_

    def fetch_batch(self):
        indices = np.random.choice(len(self.memories), self.batch_size, replace=False)
        indices = list(indices)

        for idx in indices:
            yield self.memories[idx]

    def store_trajectory(self, trajectory):
        self.memories.append(trajectory)

    def learn(self):
        if len(self.memories) < self.batch_size:
            return 

        losses = []
        for memory_idx, memory in enumerate(self.fetch_batch()):
            states, actions, rewards, dones = memory.fetch_on_device(self.device)

            self.net.train()

            hidden_state = self.net.get_new_hidden_state().to(self.device)
            second_to_last_memory_index = len(memory.states) - 1
            for i in range(second_to_last_memory_index):
                state   = states[i].detach()
                state_  = states[i+1].detach()
                action  = actions[i].detach()
                reward  = rewards[i].detach()
                
                if i == second_to_last_memory_index:
                    done = True
                else:
                    done = False

                qs, hidden_state_ = self.net(state, hidden_state)
                chosen_q = qs[action]

                qs_, hidden_state_3 = self.target_net(state_, hidden_state_)
                action_qs_, hidden_state_3 = self.net(state_, hidden_state_)
                action_ = torch.argmax(action_qs_)
                chosen_q_ = qs_[action_]
                if done:
                    chosen_q_ *= 0.0

                q_target = reward + self.gamma * chosen_q_

                loss = (q_target -  chosen_q) ** 2

                losses.append(loss)

                hidden_state = hidden_state_

        all_losses = sum(losses)
        self.optimizer.zero_grad()
        all_losses.backward()
        self.optimizer.step()

        for i in range(self.batch_size):
            self.epsilon.step()
        self.update_target_net_params()