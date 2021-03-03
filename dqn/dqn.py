import math
import random

import gym       
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ReplayBuffer:
    def __init__(self, size, state_shape):
        self.size = size
        self.count = 0

        self.state_memory       = np.zeros((self.size, *state_shape), dtype=np.float32)
        self.action_memory      = np.zeros( self.size,                dtype=np.int64  )
        self.reward_memory      = np.zeros( self.size,                dtype=np.float32)
        self.next_state_memory  = np.zeros((self.size, *state_shape), dtype=np.float32)
        self.done_memory        = np.zeros( self.size,                dtype=np.bool   )

    def store_memory(self, state, action, reward, next_state, done):
        index = self.count % self.size 
        
        self.state_memory[index]      = state
        self.action_memory[index]     = action
        self.reward_memory[index]     = reward
        self.next_state_memory[index] = next_state
        self.done_memory[index]       = done

        self.count += 1

    def sample(self, sample_size):
        highest_index = min(self.count, self.size)
        indices = np.random.choice(highest_index, sample_size, replace=False)

        states      = self.state_memory[indices]
        actions     = self.action_memory[indices]
        rewards     = self.reward_memory[indices]
        next_states = self.next_state_memory[indices]
        dones       = self.done_memory[indices]

        return states, actions, rewards, next_states, dones

class Network(torch.nn.Module):
    def __init__(self, learn_rate, input_shape, num_actions):
        super().__init__()
        self.fc1_dims = 1024
        self.fc2_dims = 512

        self.fc1 = nn.Linear(*input_shape,  self.fc1_dims)
        self.fc2 = nn.Linear( self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear( self.fc2_dims, num_actions  )

        self.optimizer = optim.Adam(self.parameters(), lr=learn_rate)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

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
    def __init__(self, learn_rate, input_shape, num_actions, batch_size):
        self.net = Network(learn_rate, input_shape, num_actions)
        self.memory = ReplayBuffer(size=100000, state_shape=input_shape)
        self.epsilon = LinearSchedule(start=1.0, end=0.01, num_steps=2000)
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.gamma = 0.99

    def choose_action(self, observation):
        if random.random() > self.epsilon.value():
            state = torch.tensor(observation).float().detach()
            state = state.to(self.net.device)
            state = state.unsqueeze(0)

            q_values = self.net(state)
            action = torch.argmax(q_values).item()

            return action
        else:
            action = random.randint(0, self.num_actions - 1)
            return action

    def store_memory(self, state, action, reward, state_, done):
        self.memory.store_memory(state, action, reward, state_, done)

    def learn(self):
        if self.memory.count < self.batch_size:
            return

        self.net.optimizer.zero_grad()
    
        states, actions, rewards, states_, dones = \
            self.memory.sample(self.batch_size)
        states  = torch.tensor( states  ).to(self.net.device)
        actions = torch.tensor( actions ).to(self.net.device)
        rewards = torch.tensor( rewards ).to(self.net.device)
        states_ = torch.tensor( states_ ).to(self.net.device)
        dones   = torch.tensor( dones   ).to(self.net.device)
        
        batch_indices = np.arange(self.batch_size, dtype=np.int64)

        action_qs = self.net(states)[batch_indices, actions]    #   (batch_size, 1)

        all_qs_ =   self.net(states_)               #   (batch_size, num_actions)
        action_qs_ = torch.max(all_qs_, dim=1)[0]   #   (batch_size, 1)
        action_qs_[dones] = 0.0

        q_targets = rewards + self.gamma * action_qs_

        loss = self.net.loss(q_targets, action_qs).to(self.net.device)
        loss.backward()
        self.net.optimizer.step()

        self.epsilon.step()

if __name__ == '__main__':
    env = gym.make('CartPole-v1').unwrapped
    agent = Agent(learn_rate=0.001, input_shape=(4,), num_actions=2, batch_size=64)

    high_score = -math.inf
    episode = 0
    num_samples = 0
    while True:
        done = False
        state = env.reset()

        score, frame = 0, 1
        while not done:
            env.render()

            action = agent.choose_action(state)
            state_, reward, done, info = env.step(action)
            agent.store_memory(state, action, reward, state_, done)
            agent.learn()

            state = state_

            num_samples += 1
            score += reward
            frame += 1

        high_score = max(high_score, score)

        print(("total samples: {}, ep {}: high-score {:12.3f}, score {:12.3f}").format(
            num_samples, episode, high_score, score, frame))

        episode += 1