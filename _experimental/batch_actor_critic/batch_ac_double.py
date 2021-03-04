import math
import random

import gym       
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

'''
        --  Batch Double Actor Critic --
        Batch Actor Critic has problems with catastrophic forgetting. 
        I am unsure as to weather this is instability of the policy gradient, 
        or the value estimations. Double should fix instability in the value estimations, 
        but does not address the policy gradient noise.

        Goal:
        -fix instability in the value estimations.

        Performance:
        -stable as a rock, 
            that is also positioned on something stable, 
                such as another rock.

        Flaws:
        -slow start.
            1. either the value net is warming up due to the delayed updates 
                    or
            2. the policy gradient is still too noisy

        Benefit:
        Nice it works. Batch Actor Critic is great with a target value network, 
            even if the method of updating is sudden/primitive. 

        Potential:
        -needs a target policy to stabilize policy gradient
        -maaybe soft updates

        Output Sample:
        >>>
            total samples: 19245, ep 152: high-score      840.000, score      213.000
            total samples: 19467, ep 153: high-score      840.000, score      222.000
            total samples: 19729, ep 154: high-score      840.000, score      262.000
            total samples: 20221, ep 155: high-score      840.000, score      492.000
            total samples: 20748, ep 156: high-score      840.000, score      527.000
            total samples: 21378, ep 157: high-score      840.000, score      630.000
            total samples: 22176, ep 158: high-score      840.000, score      798.000
            total samples: 22569, ep 159: high-score      840.000, score      393.000
            total samples: 23155, ep 160: high-score      840.000, score      586.000
            total samples: 24259, ep 162: high-score      840.000, score      485.000
'''

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

    def get_values(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.critic(x)

        return value

    def get_policy(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy = self.actor(x)

        return policy

class Agent():
    def __init__(self, learn_rate, input_shape, num_actions, batch_size):
        self.net = Network(learn_rate, input_shape, num_actions)
        self.target_net = Network(learn_rate, input_shape, num_actions)

        self.memory = ReplayBuffer(size=100000, state_shape=input_shape)
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.gamma = 0.99

        self.learn_step_counter = 0
        self.net_copy_interval = 10

    def choose_action(self, observation):
        state = torch.tensor(observation).float().detach()
        state = state.to(self.net.device)
        state = state.unsqueeze(0)
        
        self.net.eval()
        policy = self.net.get_policy(state)

        policy = policy[0]
        policy = F.softmax(policy, dim=0)
        actions_probs = torch.distributions.Categorical(policy)
        action = actions_probs.sample()
        self.action_log_prob = actions_probs.log_prob(action)
        return action.item()

    def get_log_probs(self, states, actions, target=False):
        if not target:
            self.net.train()
            policy = self.net.get_policy(states)
        else:
            self.target_net.train()
            policy = self.target_net.get_policy(states)

        policy = F.softmax(policy, dim=1)
        actions_dist = torch.distributions.Categorical(policy)
        action_log_probs = actions_dist.log_prob(actions)
        action_log_probs = action_log_probs.unsqueeze(0).T
        return action_log_probs

    def store_memory(self, state, action, reward, state_, done):
        self.memory.store_memory(state, action, reward, state_, done)

    def update_params(self):
        net_params= self.net.named_parameters()
        target_net_params = self.target_net.named_parameters()

        target_net = dict(net_params)
        net_ = dict(target_net_params)

        for name in target_net:
            target_net[name] = tau * target_net[name].clone() + (1 - tau) * net_[name].clone()

        self.target_net.load_state_dict(target_net)

    def learn(self):
        if self.memory.count < self.batch_size:
            return

        self.net.train()
        # self.target_net.train()
        self.net.optimizer.zero_grad()

        states, actions, rewards, states_, dones = \
            self.memory.sample(self.batch_size)
        states  = torch.tensor( states  ).to(self.net.device)
        actions = torch.tensor( actions ).to(self.net.device)
        rewards = torch.tensor( rewards ).to(self.net.device)
        states_ = torch.tensor( states_ ).to(self.net.device)
        dones   = torch.tensor( dones   ).to(self.net.device)
        
        values  = self.net.get_values(states)
        values_ = self.target_net.get_values(states_)
        values_[dones] = 0.0

        targets = rewards + self.gamma * values_
        td = targets - values

        log_probs = self.get_log_probs(states, actions)

        actor_loss = (-log_probs * td).mean()
        critic_loss = (td**2).mean()

        loss = actor_loss + critic_loss
        loss.backward()
        self.net.optimizer.step()

        if self.learn_step_counter % self.net_copy_interval == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        self.learn_step_counter += 1

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