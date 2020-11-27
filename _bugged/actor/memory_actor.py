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
    def __init__(self, learn_rate, input_shape, num_actions, device):
        super().__init__()
        self.fc1_dims = 1024
        self.fc2_dims = 512
        self.device = device

        self.fc1 = nn.Linear(*input_shape,  self.fc1_dims)
        self.fc2 = nn.Linear( self.fc1_dims, self.fc2_dims)
        self.fc3  = nn.Linear( self.fc2_dims, num_actions  )

        self.optimizer = optim.Adam(self.parameters(), lr=learn_rate)
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        urge = self.fc3(x)

        return urge

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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")

        self.net = Network(learn_rate, input_shape, num_actions, self.device)
        self.memory = ReplayBuffer(size=100000, state_shape=input_shape)
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.gamma = 0.99
        self.epsilon = LinearSchedule(start=0.0, end=0.0, num_steps=2000)

    def choose_action(self, observation):
        if random.random() > self.epsilon.value():
            state = torch.tensor(observation).float().detach()
            state = state.to(self.net.device)
            state = state.unsqueeze(0)

            action_urges = self.net(state)
            # print(action_urges)
            action = torch.argmax(action_urges).item()

            return action
        else:
            action = random.randint(0, self.num_actions - 1)
            return action

    def store_memory(self, state, action, reward, state_, done):
        self.memory.store_memory(state, action, reward, state_, done)

    def learn(self, punishment):
        if self.memory.count < self.batch_size:
            return

        states, actions, rewards, states_, dones = \
            self.memory.sample(self.batch_size)
        states  = torch.tensor( states  ).to(self.device)
        actions = torch.tensor( actions ).to(self.device)
        rewards = torch.tensor( rewards ).to(self.device)
        states_ = torch.tensor( states_ ).to(self.device)
        dones   = torch.tensor( dones   ).to(self.device)

        batch_indices = torch.arange(self.batch_size, dtype=torch.int64).to(self.device)

        self.net.train()
        urges = self.net(states)

        targets = torch.zeros(
            (self.batch_size, self.num_actions), 
            dtype=torch.float32).to(self.device)
        punishment = torch.tensor(punishment, dtype=torch.float32).to(self.device)
        targets[batch_indices, action] -= punishment

        self.net.optimizer.zero_grad()
        loss = ((targets - urges)**2).mean()
        loss.backward()
        self.net.optimizer.step()

        self.epsilon.step()

if __name__ == '__main__':
    env = gym.make('CartPole-v1').unwrapped
    BATCH_SIZE = 64
    agent = Agent(learn_rate=0.0001, input_shape=(4,), num_actions=2, batch_size=BATCH_SIZE)

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
            if done:
                agent.store_memory(state, action, reward, state_, done)
            agent.learn(punishment=1.0)

            state = state_

            score += reward
            frame += 1
        if ep >= BATCH_SIZE:
            num_samples += 1

        high_score = max(high_score, score)

        print(("total samples: {}, ep {}: high-score {:12.3f}, score {:12.3f}, eps {:12.3f}").format(
            num_samples, episode, high_score, score, agent.epsilon.value()))

        episode += 1