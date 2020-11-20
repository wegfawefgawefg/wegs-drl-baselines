import math
import random

import gym       
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Network(torch.nn.Module):
    def __init__(self, learn_rate, input_shape, num_actions):
        super().__init__()
        self.fc1_dims = 1024
        self.fc2_dims = 512

        self.fc1 = nn.Linear(*input_shape,  self.fc1_dims)
        self.fc2 = nn.Linear( self.fc1_dims, self.fc2_dims)
        self.fc3  = nn.Linear( self.fc2_dims, num_actions  )

        self.optimizer = optim.Adam(self.parameters(), lr=learn_rate)
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
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
        self.net = Network(learn_rate, input_shape, num_actions)
        self.num_actions = num_actions
        self.gamma = 0.99
        self.epsilon = LinearSchedule(start=1.0, end=0.01, num_steps=2000)

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

    def learn(self, state, action, reward, state_, done, punishment):
        self.net.train()
    
        state = torch.tensor(state).float()
        state = state.to(self.net.device)
        state = state.unsqueeze(0)

        urge  = self.net(state)

        target = torch.zeros((1, self.num_actions), dtype=torch.float32)
        # target = urge.clone().detach()
        if done:
            target[0, action] -= self.epsilon.value()

        self.net.optimizer.zero_grad()
        loss = F.mse_loss(target, urge)
        loss.backward()
        self.net.optimizer.step()

        self.epsilon.step()

if __name__ == '__main__':
    env = gym.make('CartPole-v1').unwrapped
    agent = Agent(learn_rate=0.0001, input_shape=(4,), num_actions=2, batch_size=64)

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
            agent.learn(state, action, reward, state_, done, 0.1)

            state = state_

            num_samples += 1
            score += reward
            frame += 1

        high_score = max(high_score, score)

        print(("total samples: {}, ep {}: high-score {:12.3f}, score {:12.3f}").format(
            num_samples, episode, high_score, score, frame))

        episode += 1