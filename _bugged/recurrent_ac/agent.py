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
        self.gamma = 0.99
        self.critic_update_max = 20
        self.actor_update_max = 10
        self.memories = []

        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

        self.actor = Actor().to(self.device)
        self.critic = Critic().to(self.device)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learn_rate)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learn_rate)

    def choose_action(self, state, hidden_state):
        state = torch.tensor(state,        dtype=torch.float32).to(self.device)

        policy, hidden_state_ = self.actor(state, hidden_state)
        policy = F.softmax(policy)
        actions_probs = torch.distributions.Categorical(policy)
        action = actions_probs.sample()
        action_log_prob = actions_probs.log_prob(action).unsqueeze(0)
        # action = torch.argmax(policy)
        

        #   prep for storage
        action = action.item()

        return action, policy, hidden_state_, action_log_prob

    def store_memory(self, memory):
        self.memories.append(memory)

    def get_discounted_cum_rewards(self, memory):
        cum_rewards = []
        total = 0
        for reward in reversed(memory.rewards):
            total = reward + total * self.gamma
            cum_rewards.append(total)
        cum_rewards = list(reversed(cum_rewards))
        cum_disc_rewards = torch.tensor(cum_rewards).float().to(self.device)
        
        return cum_rewards

    def learn(self):
        critic_losses = []
        for memory_idx, memory in enumerate(self.memories):
            print(memory_idx)
            states, actions, policies, rewards, dones, actor_hidden_states, action_log_probs = \
                memory.fetch_on_device(self.device)
            cum_disc_rewards = self.get_discounted_cum_rewards(memory)
            ''' train critic    '''
            self.critic.train()
            self.actor.eval()

            critic_hidden_state = self.critic.get_new_hidden_state()
            for i in range(len(memory.states)):
                state = states[i].detach()
                policy = policies[i].detach()
                action_log_prob = action_log_probs[i].detach()
                done = dones[i].detach()

                true_value = cum_disc_rewards[i]

                value, critic_hidden_state_ = self.critic(state, action_log_prob, critic_hidden_state)
                if done:
                    true_value *= 0.0
                error = value - true_value
                # print("true: {}, value: {}".format(true_value, value))
                critic_loss = error**2
                if critic_loss >= self.critic_update_max:
                    print("critic_loss BIG: {}".format(critic_loss))
                critic_loss = torch.clamp(critic_loss, -self.critic_update_max, self.critic_update_max)
                critic_losses.append(critic_loss)

                critic_hidden_state = critic_hidden_state_

        # print("end")
        all_critic_loss = sum(critic_losses)
        # all_critic_loss = torch.stack(critic_losses).mean()
        self.critic_optimizer.zero_grad()
        all_critic_loss.backward()
        self.critic_optimizer.step()

        actor_losses = []
        for memory_idx, memory in enumerate(self.memories):
            print(memory_idx)
            states, actions, policies, rewards, dones, actor_hidden_states, action_log_probs = \
                memory.fetch_on_device(self.device)            
            ''' train actor     '''
            self.critic.eval()
            self.actor.train()

            critic_hidden_state = self.critic.get_new_hidden_state()
            for i in range(len(memory.states)):
                state = states[i].detach()
                # policy = policies[i]
                action_log_prob = action_log_probs[i]
                critic_hidden_state = critic_hidden_state.detach()
                done = dones[i].detach()

                value, critic_hidden_state_ = self.critic(state, action_log_prob, critic_hidden_state)
                if done:
                    value *= 0.0
                # print("true: {}, value: {}".format(true_value, value))
                actor_loss = value
                if actor_loss >= self.actor_update_max:
                    print("actor_loss BIG: {}".format(actor_loss))
                actor_loss = torch.clamp(actor_loss, -self.actor_update_max, self.actor_update_max)
                actor_losses.append(actor_loss)

                critic_hidden_state = critic_hidden_state_

        all_actor_loss = sum(actor_losses)
        # all_actor_loss = torch.stack(actor_losses).mean()
        self.actor_optimizer.zero_grad()
        all_actor_loss.backward()
        self.actor_optimizer.step()
