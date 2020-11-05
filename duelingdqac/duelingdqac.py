import math
import random

import gym       
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

'''
        --  Dueling Deep Q Actor Critic --
        
        Vanilla actor critic, except add dueling dqn to the value estimation.
        -   or  -
        Dueling dqn, but with a non greedy online policy gradient actor.

        Goal:
        Essentially, we are removing some burdern from the value estimator.
        1. the value estimator no longer has to guess the value of the specific actions themselves
        2. the value estimator is no longer burdened with having to predict which action the agent will take next

        Additional Potential:
        I feel like actor should get additional information from the new calculated values.
        It would stabilize the log prob updates.
        -perhaps from comparing future and now log probs, the loss could be clipped using a running average
        -perhaps from using an additional network to predict the direction log probs will go, 
            using the difference between the past and future prob as a delta (meta gradient)
        -perhaps an addition log prob modification from the next action
        -perhaps some sort of mean log prob fuckery (true advantage actor),
            by taking the mean probability of all the actions, 
            and only using the deviation of the chosen actions probability from the mean

        Performance:

'''

class Network(torch.nn.Module):
    def __init__(self, learn_rate, input_shape, num_actions):
        super().__init__()
        self.fc1_dims = 1024
        self.fc2_dims = 512

        self.fc1 = nn.Linear(*input_shape,  self.fc1_dims)
        self.fc2 = nn.Linear( self.fc1_dims, self.fc2_dims)

        self.critic = nn.Linear( self.fc2_dims, 1)
        self.actor  = nn.Linear( self.fc2_dims, num_actions  )
        self.q      = nn.Linear( self.fc2_dims, num_actions  )

        self.optimizer = optim.Adam(self.parameters(), lr=learn_rate)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value   = self.critic(x)
        policy  = self.actor(x)
        qs      = self.q(x) 

        return policy, value, qs

    def get_policy(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy = self.actor(x)

        return policy

    def get_values_and_qs(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.critic(x)
        qs = self.q(x)

        return value, qs

class Agent():
    def __init__(self, learn_rate, input_shape, num_actions, batch_size):
        self.net = Network(learn_rate, input_shape, num_actions)
        self.num_actions = num_actions
        self.gamma = 0.99

    def choose_action(self, observation, next=False):
        self.net.eval()

        state = torch.tensor(observation).float().detach()
        state = state.to(self.net.device)
        state = state.unsqueeze(0)
        
        policy = self.net.get_policy(state)

        policy = policy[0]
        policy = F.softmax(policy, dim=0)
        actions_probs = torch.distributions.Categorical(policy)
        action = actions_probs.sample()
        if not next:
            self.action_log_prob = actions_probs.log_prob(action)
        else:   #   not sure why you would need this but maybe it has some useful information
            self.next_action_log_prob = actions_probs.log_prob(action)
        return action.item()

    def calc_advantages(self, qs):
        return qs - qs.mean(dim=1)

    def learn(self, state, action, reward, state_, done):
        self.net.train()
        self.net.optimizer.zero_grad()
    
        state = torch.tensor(state).float()
        state = state.to(self.net.device)
        state = state.unsqueeze(0)

        state_ = torch.tensor(state_).float()
        state_ = state_.to(self.net.device)
        state_ = state_.unsqueeze(0)

        action = torch.tensor(reward, dtype=torch.int64 ).to(self.net.device)
        reward = torch.tensor(reward, dtype=torch.float ).to(self.net.device)
        done   = torch.tensor(done,   dtype=torch.bool  ).to(self.net.device)

        value, qs  = self.net.get_values_and_qs(state)
        value_, qs_ = self.net.get_values_and_qs(state_)
        value_[done] = 0.0
        qs_[done] = 0.0

        qs_adv = self.calc_advantages(qs)
        qs_adv_ = self.calc_advantages(qs_)
        action_ = self.choose_action(state_, next=True)

        past = value + qs_adv[0, action]
        future = value_ + qs_adv_[0, action_]

        target = reward + self.gamma * future
        td = target - past


        actor_loss = -self.action_log_prob * td 
        critic_and_q_loss = td**2

        loss = actor_loss + critic_and_q_loss
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
            agent.learn(state, action, reward, state_, done)

            state = state_

            num_samples += 1
            score += reward
            frame += 1

        high_score = max(high_score, score)

        print(("total samples: {}, ep {}: high-score {:12.3f}, score {:12.3f}").format(
            num_samples, episode, high_score, score, frame))

        episode += 1