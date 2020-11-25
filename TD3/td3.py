import math
import random
import copy 

import gym       
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from continuous_cartpole import ContinuousCartPoleEnv

'''
    Performance:
        -very stable and good scores once its going
        -slow start. 
            have to wait for the critic to mature.
        -medium run speed

    Output Sample:
    >>>
        total samples: 59721, ep 385: high-score     8299.000, score     8299.000
        total samples: 61101, ep 386: high-score     8299.000, score     1380.000
        total samples: 62775, ep 387: high-score     8299.000, score     1674.000
        total samples: 63330, ep 388: high-score     8299.000, score      555.000
        total samples: 63933, ep 389: high-score     8299.000, score      603.000
        total samples: 64352, ep 390: high-score     8299.000, score      419.000
        total samples: 64888, ep 391: high-score     8299.000, score      536.000
        total samples: 88121, ep 392: high-score    23233.000, score    23233.000
'''

class ReplayBuffer:
    def __init__(self, size, state_shape, num_actions):
        self.size = size
        self.count = 0

        self.state_memory       = np.zeros((self.size, *state_shape ), dtype=np.float32)
        self.action_memory      = np.zeros((self.size, num_actions  ), dtype=np.float32)
        self.reward_memory      = np.zeros((self.size, 1            ), dtype=np.float32)
        self.next_state_memory  = np.zeros((self.size, *state_shape ), dtype=np.float32)
        self.done_memory        = np.zeros((self.size, 1            ), dtype=np.bool   )

    def store_memory(self, state, action, reward, next_state, done):
        index = self.count % self.size 
        
        self.state_memory[index]      = state
        self.action_memory[index]     = action
        self.reward_memory[index]     = reward
        self.next_state_memory[index] = next_state
        self.done_memory[index]       = done

        self.count += 1

    def sample(self, sample_size, device):
        highest_index = min(self.count, self.size)
        indices = np.random.choice(highest_index, sample_size, replace=False)

        states  = self.state_memory[indices]
        actions = self.action_memory[indices]
        rewards = self.reward_memory[indices]
        states_ = self.next_state_memory[indices]
        dones   = self.done_memory[indices]

        states  = torch.tensor( states  ).to(device)
        actions = torch.tensor( actions ).to(device)
        rewards = torch.tensor( rewards ).to(device)
        states_ = torch.tensor( states_ ).to(device)
        dones   = torch.tensor( dones   ).to(device)

        return states, actions, rewards, states_, dones

class Actor(torch.nn.Module):
    def __init__(self, layer_sizes, state_shape, num_actions, max_action):
        super().__init__()
        self.max_action = max_action
        self.model = nn.Sequential(
            nn.Linear(*state_shape,    layer_sizes[0]),     nn.ReLU(),
            nn.Linear( layer_sizes[0], layer_sizes[1]),     nn.ReLU(),
            nn.Linear( layer_sizes[1], num_actions))

    def forward(self, x):
        x = self.model(x)
        actions = torch.tanh(x)
        actions = self.max_action * actions
        return actions

class Critic(torch.nn.Module):
    def __init__(self, layer_sizes, state_shape, num_actions):
        super().__init__()
        self.fc1_dims = layer_sizes[0]
        self.fc2_dims = layer_sizes[1]

        self.q1 = nn.Sequential(
            nn.Linear( state_shape[0] + num_actions,    layer_sizes[0]),     nn.ReLU(),
            nn.Linear( layer_sizes[0],                  layer_sizes[1]),     nn.ReLU(),
            nn.Linear( layer_sizes[1], num_actions))

        self.q2 = nn.Sequential(
            nn.Linear( state_shape[0] + num_actions,    layer_sizes[0]),     nn.ReLU(),
            nn.Linear( layer_sizes[0],                  layer_sizes[1]),     nn.ReLU(),
            nn.Linear( layer_sizes[1], num_actions))

    def forward(self, states, actions):
        states_actions = torch.cat([states, actions], 1)
        q1_values = self.q1(states_actions)
        q2_values = self.q2(states_actions)

        return q1_values, q2_values

    def get_critic_one_values(self, states, actions):
        states_actions = torch.cat([states, actions], 1)
        values = self.q1(states_actions)
        return values

class Agent(object):
    def __init__(self,
            #   basics
            state_shape,
            num_actions,
            max_action,
            layer_sizes=(256, 256), 
            batch_size=64,              #Paper:    100,
            gamma=0.99,

            #   replay buffer
            buffer_size=1_000_000,
            min_buffer_fullness=64,     #Paper: 10_000,

            #   learn rates
            critic_learn_rate=3e-4,    
            actor_learn_rate=3e-4,     
            tau=0.005,

            #   update frequencies / delays
            num_babbling_steps=100,     #Paper: 10_000,
            critic_maturity_delay=100,  #Paper: 10_000,
            critic_update_freq=1,       #Paper:     50,
            actor_update_freq=2,

            #   noise
            policy_noise=0.2,
            noise_clamp=0.5,
            ):

        '''   SETTINGS   '''
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu,")
        self.device = torch.device("cpu")

        self.num_actions = num_actions
        self.max_action = max_action
        self.batch_size = batch_size
        self.gamma      = gamma

        self.min_buffer_fullness = min_buffer_fullness

        self.tau = tau
        
        self.num_babbling_steps = num_babbling_steps
        self.critic_maturity_delay = critic_maturity_delay
        self.critic_update_freq = critic_update_freq
        self.actor_update_freq = actor_update_freq

        self.policy_noise = policy_noise
        self.noise_clamp = noise_clamp

        '''   STATE   '''
        self.num_actions_taken = 0
        self.learn_calls = 0
        self.num_critic_learn_steps = 0
        self.num_actor_learn_steps = 0

        self.memory_minimum_fullness_announced == False
        self.critic_maturity_announced == False

        self.memory = ReplayBuffer(buffer_size, state_shape, num_actions)

        self.actor  = Actor( layer_sizes, state_shape, num_actions, max_action  ).to(self.device)
        self.critic = Critic(layer_sizes, state_shape, num_actions              ).to(self.device)

        self.actor_target  = copy.deepcopy(self.actor ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(),  lr=actor_learn_rate )
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_learn_rate)

    def store_memory(self, state, actions, reward, state_, done):
        self.memory.store_memory(state, actions, reward, state_, done)

    def choose_action(self, state, deterministic=False):
        self.num_actions_taken += 1
        if self.num_actions_taken == self.num_babbling_steps:
            print("Action babbling finished.")
        elif self.num_actions_taken < self.num_babbling_steps:
            actions = np.random.uniform(low=-self.max_action, high=self.max_action, size=self.num_actions)
            return actions
        
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state)
        if not deterministic:
            noise = torch.randn_like(action) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clamp, self.noise_clamp)
            action = action + noise
            action = action.clamp(-self.max_action, self.max_action)
        action = action.cpu().data.numpy().flatten()

        return action

    def update_target_critic_params(self):
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update_target_actor_params(self):
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update_critic(self, data):
        states, actions, rewards, states_, dones = data
        with torch.no_grad():
            actions_ = self.actor_target(states_)
            noise = torch.randn_like(actions) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clamp, self.noise_clamp)
            actions_ = actions_ + noise
            actions_ = actions_.clamp(-self.max_action, self.max_action)

            values_one_, values_two_ = self.critic_target(states_, actions_)
            
            values_ = torch.min(values_one_, values_two_)
            values_[dones] = 0.0

            target = rewards + self.gamma * values_

        values_one, values_two = self.critic(states, actions)
        critic_one_loss = F.mse_loss(values_one, target)
        critic_two_loss = F.mse_loss(values_two, target)
        critic_loss = critic_one_loss + critic_two_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor(self, data):
        states, actions, rewards, states_, dones = data
        actor_loss = -self.critic.get_critic_one_values(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def learn(self):
        if self.memory.count < self.min_buffer_fullness:
            return
        elif not self.memory_minimum_fullness_announced:
            print("Memory reached minimum fullness.")
            self.memory_minimum_fullness_announced == True

        self.learn_calls += 1

        update_critic = self.learn_calls % self.critic_update_freq == 0
        update_actor = self.learn_calls % self.actor_update_freq == 0
        if update_critic or update_actor:
            data = self.memory.sample(self.batch_size, self.device)
            if update_critic:
                self.update_critic(data)
                self.update_target_critic_params()
                self.num_critic_learn_steps += 1
            if update_actor:
                if self.num_critic_learn_steps <= self.critic_maturity_delay:
                    return
                elif not self.critic_maturity_announced:
                    print("Critic values deemed mature.")
                    self.critic_maturity_announced == True

                self.update_actor(data)
                self.update_target_actor_params()
                self.num_actor_learn_steps += 1
        
if __name__ == '__main__':
    env = ContinuousCartPoleEnv()
    # env = gym.make("LunarLanderContinuous-v2")
    agent = Agent(state_shape=(4,), num_actions=1, max_action=1.0)

    high_score = -math.inf
    episode = 0
    num_samples = 0
    while True:
        done = False
        state = env.reset()

        score, frame = 0, 1
        while not done:
            env.render()

            actions = agent.choose_action(state)
            state_, reward, done, info = env.step(actions)
            agent.store_memory(state, actions, reward, state_, done)
            agent.learn()

            state = state_

            num_samples += 1
            score += reward
            frame += 1

        high_score = max(high_score, score)

        print(("total samples: {}, ep {}: high-score {:12.3f}, score {:12.3f}").format(
            num_samples, episode, high_score, score, frame))

        episode += 1