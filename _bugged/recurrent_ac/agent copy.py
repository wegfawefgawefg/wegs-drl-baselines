import math
import random

from pprint import pprint
import gym       
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models import Actor, Critic
from memory import Memory

class Agent():
    def __init__(self, learn_rate, input_shape, num_actions):
        self.num_actions = num_actions
        self.gamma = 0.9999
        self.memory = Memory()

        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

        self.actor = Actor().to(self.device)
        self.critic = Critic().to(self.device)

        critic_params = list(self.critic.parameters())
        pprint(critic_params)
        quit()

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learn_rate)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learn_rate)

    def choose_action(self, state, hidden_state):
        state = torch.tensor(state,        dtype=torch.float32).to(self.device)

        policy, hidden_state_ = self.actor(state, hidden_state)
        action = torch.argmax(policy)

        #   prep for storage
        action = action.item()

        return action, policy, hidden_state_

    def store_memory(self, memory):
        self.memory.store(*memory)

    def get_discounted_cum_rewards(self):
        cum_rewards = []
        total = 0
        for reward in reversed(self.memory.rewards):
            total = reward + total * self.gamma
            cum_rewards.append(total)
        cum_rewards = list(reversed(cum_rewards))
        cum_disc_rewards = torch.tensor(cum_rewards).float().to(self.device)
        
        return cum_rewards

    def learn(self):
        states, actions, policies, rewards, dones, actor_hidden_states = self.memory.fetch_on_device(self.device)
        cum_disc_rewards = self.get_discounted_cum_rewards()
        # print("cum_disc_rewards")
        # pprint(cum_disc_rewards)
        # quit()

        ''' train critic    '''
        self.critic.train()
        self.actor.eval()

        critic_losses = []
        critic_hidden_state = self.critic.get_new_hidden_state()
        for i in range(len(self.memory.states) - 1):
            state = states[i].detach()
            policy = policies[i].detach()
            true_value = cum_disc_rewards[i]

            value, critic_hidden_state_ = self.critic(state, policy, critic_hidden_state)
            error = value - true_value
            # print("true: {}, value: {}".format(true_value, value))
            critic_loss = error**2
            critic_losses.append(critic_loss)

            critic_hidden_state = critic_hidden_state_

        # print("end")
        all_critic_loss = sum(critic_losses)
        self.critic_optimizer.zero_grad()
        all_critic_loss.backward()
        self.critic_optimizer.step()

        ''' train actor     '''
        self.critic.eval()
        self.actor.train()

        actor_losses = []
        critic_hidden_state = self.critic.get_new_hidden_state()
        for i in range(len(self.memory.states) - 1):
            state = states[i].detach()
            policy = policies[i]
            critic_hidden_state = critic_hidden_state.detach()

            value, critic_hidden_state_ = self.critic(state, policy, critic_hidden_state)
            # print("true: {}, value: {}".format(true_value, value))
            actor_loss = value
            actor_losses.append(actor_loss)

            critic_hidden_state = critic_hidden_state_

        all_actor_loss = sum(actor_losses)
        self.actor_optimizer.zero_grad()
        all_actor_loss.backward()
        self.actor_optimizer.step()

    def learn_old(self):
        self.actor.eval()
        self.critic.train()
    
        state = torch.tensor(state).float()
        state = state.to(self.device)
        state_ = torch.tensor(state_).float()
        state_ = state_.to(self.device)

        reward = torch.tensor(reward, dtype=torch.float).to(self.device)
        done = torch.tensor(done, dtype=torch.bool).to(self.device)

        value, critic_hidden_state_ = self.critic(state, policy, self.critic_hidden_state)

        policy_, _ = self.actor(state, self.actor_hidden_state)
        value_, critic_hidden_state_ = self.critic(state_, policy_, critic_hidden_state_)
        if done:
            value_ = 0.0

        target = reward + self.gamma * value_
        td = target - value

        critic_loss = td**2

        self.critic_optimizer.zero_grad()
        if not done:
            critic_loss.backward(retain_graph=True, allow_unreachable=True)
        else:
            critic_loss.backward(allow_unreachable=True)
        self.critic_optimizer.step()

        ''' update based on new policy of old states '''
        self.critic.eval()
        self.actor.train()
        retro_value, retro_critic_hidden_state_ = self.critic(state, policy, self.critic_hidden_state)

        retro_policy_, actor_hidden_state_ = self.actor(state, self.actor_hidden_state)
        retro_value_, _ = self.critic(state_, retro_policy_, retro_critic_hidden_state_)
        if done:
            retro_value_ = 0.0

        actor_loss = -(retro_value_ - retro_value)

        self.actor_optimizer.zero_grad()
        if not done:
            actor_loss.backward(retain_graph=True, allow_unreachable=True)
        else:
            actor_loss.backward(allow_unreachable=True)
        self.actor_optimizer.step()

        ''' update hidden states    '''
        self.actor_hidden_state = actor_hidden_state_
        self.critic_hidden_state_ = critic_hidden_state_
