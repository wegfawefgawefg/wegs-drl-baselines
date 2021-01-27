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
from memory_slice_replay_buffer import MemorySliceReplayBuffer

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
    def __init__(self, learn_rate, 
            state_shape, num_actions, action_shape, 
            batch_size, slice_size):
        self.gamma = 0.999
        self.tau = 0.01
        self.clip_grad_norm = 0.1
        self.has_target_net = True

        self.state_shape = state_shape
        self.num_actions = num_actions      #   this is how many actions there are to choose from
        self.action_shape = action_shape    #   this is how many actions the env accepts at each step

        self.buffer_size = 1_000_000
        self.batch_size = batch_size    # *times slice_size, because recurrency/rollouts
        self.slice_size = slice_size

        self.slice_replay_buffer = MemorySliceReplayBuffer(
            size=self.buffer_size, slice_size=self.slice_size, 
            state_shape=self.state_shape, action_shape=self.action_shape)
        self.epsilon = LinearSchedule(start=1.0, end=0.01, num_steps=300)
        # self.epsilon = LinearSchedule(start=1.0, end=0.1, num_steps=30)


        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

        self.net = DQN(state_shape, num_actions).to(self.device)
        if self.has_target_net:
            self.target_net  = copy.deepcopy(self.net).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learn_rate)

    def update_target_net_params(self):
        for param, target_param in zip(self.net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def choose_action(self, observation, hidden_state):
            state = torch.tensor(observation).float().unsqueeze(0)
            state = state.detach().to(self.device)

            q_values, hidden_state_ = self.net(state, hidden_state)
            action = torch.argmax(q_values[0]).item()

            if random.random() <= self.epsilon.value():
                action = random.randint(0, self.action_shape[0])

            return action, hidden_state_

    def learn(self, stats):
        if self.slice_replay_buffer.count < self.batch_size:
            return 

        self.net.train()

        states_slices, actions_slices, rewards_slices, next_states_slices, dones_slices = self.slice_replay_buffer.sample(self.batch_size, self.device)

        batch_losses = []
        hidden_states = self.net.get_batch_hidden_state(self.batch_size).to(self.device)

        for slice_index in range(self.slice_size):
            states = states_slices[:, slice_index]
            actions = actions_slices[:, slice_index]
            rewards = rewards_slices[:, slice_index]
            states_ = next_states_slices[:, slice_index]
            dones = dones_slices[:, slice_index]

            batch_indices = np.arange(self.batch_size, dtype=np.int64)
            qs, hidden_states_ = self.net(states, hidden_states)
            chosen_q = qs[batch_indices, actions.T[0]]

            if self.has_target_net:
                qs_, hidden_state_3 = self.target_net(states_, hidden_states_)
                action_qs_, hidden_state_3 = self.net(states_, hidden_states_)
                actions_ = torch.argmax(action_qs_, dim=1)
                chosen_q_ = qs_[batch_indices, actions_]
            else:
                action_qs_, hidden_state_3 = self.net(states_, hidden_states_)
                chosen_q_ = torch.max(action_qs_, dim=1)[0]

            rewards = rewards.T[0]
            q_target = rewards + self.gamma * chosen_q_

            loss = torch.mean( (q_target -  chosen_q) ** 2 )
            batch_losses.append(-loss)

            hidden_states = hidden_states_
            hidden_states[dones.T[0]] = 0.0 #   if an episode ends mid slice then zero the hidden_states
                                            #   this could be a problem if backprop stops here

        batch_losses = torch.stack(batch_losses)
        batch_loss = torch.mean(batch_losses)
        stats.last_loss = batch_loss.item()
        self.optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip_grad_norm)
        self.optimizer.step()

        self.epsilon.step()
        if self.has_target_net:
            self.update_target_net_params()