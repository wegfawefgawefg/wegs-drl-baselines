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

        self.critic = nn.Linear( self.fc2_dims, 1)
        self.actor  = nn.Linear( self.fc2_dims, num_actions  )

        self.optimizer = optim.Adam(self.parameters(), lr=learn_rate)
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.critic(x)
        policy = self.actor(x)

        return value, policy

class Agent():
    def __init__(self, learn_rate, input_shape, num_actions, batch_size):
        self.net = Network(learn_rate, input_shape, num_actions)
        self.num_actions = num_actions
        self.gamma = 0.99

    def choose_action(self, observation):
        self.net.eval()

        state = torch.tensor(observation).float().detach()
        state = state.to(self.net.device)
        state = state.unsqueeze(0)
        
        _, policy = self.net(state)

        policy = policy[0]
        policy = F.softmax(policy, dim=0)
        actions_probs = torch.distributions.Categorical(policy)
        action = actions_probs.sample()
        self.action_log_prob = actions_probs.log_prob(action)
        return action.item()

    def learn(self, state, reward, state_, done):
        self.net.train()
        self.net.optimizer.zero_grad()
    
        state = torch.tensor(state).float()
        state = state.to(self.net.device)
        state = state.unsqueeze(0)

        state_ = torch.tensor(state_).float()
        state_ = state_.to(self.net.device)
        state_ = state_.unsqueeze(0)

        reward = torch.tensor(reward, dtype=torch.float).to(self.net.device)
        done = torch.tensor(done, dtype=torch.bool).to(self.net.device)

        value, _  = self.net(state)
        value_, _ = self.net(state_)
        value[done] = 0.0

        target = reward + self.gamma * value_
        td = target - value

        actor_loss = -self.action_log_prob * td
        critic_loss = td**2

        loss = actor_loss + critic_loss
        loss.backward()
        self.net.optimizer.step()

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
            agent.learn(state, reward, state_, done)

            state = state_

            num_samples += 1
            score += reward
            frame += 1

        high_score = max(high_score, score)

        print(("total samples: {}, ep {}: high-score {:12.3f}, score {:12.3f}").format(
            num_samples, episode, high_score, score, frame))

        episode += 1