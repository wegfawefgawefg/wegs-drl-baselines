import math
import random

import gym       
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from continuous_cartpole import ContinuousCartPoleEnv

'''
        --  Simpleton's Deeply Non-Deterministic Policy Gradient --
        ///////////////////////////////////////////////////////////////////////////////////////////////////
            TLDR:
            Lesson learned is that DDPG does not need the frills. 
            Works fine without noise, batchnorm, or even deterministic policy.
        ///////////////////////////////////////////////////////////////////////////////////////////////////
        
        The intention of DDPG was to bring the Q(s, a) function to continuous space. 
        But, DDPG is essentially Batch Soft-Double Actor Critic, with some noise added in for exploration. 
        Fun fact: This loss function doesnt seem to work for discrete space at all, 
        because the gradient cant shift the actions to nearby actions.
        More importantly, I don't know where the good performance comes from, and the actual version is much 
        more complicated to implement. So... I'm not going to include the noise. I've played with noise. 
        I've read the propoganda. At the moment I'm not convinced that it even helps most of the time. 
        Additionally, I dont care about the deterministic component. With a high enough tic rate 
        determinism is an emergent property. Basically this is modified loss batch actor critic.

        Goal:
        -to see if ddpg needs the frills

        Performance:
        -learns basics suddenly and quickly
        -im not impressed with its learning rate but it does eventually become reliable

        Flaws:
        -no discrete
        -seems to get stuck on actions like normal PG
        - in the early stage of getting its value/policy bearings, policy decays after being learned
            -you can see it just by watching...
            -not sure if issue with value side
            -or policy updates are too big
            -dqn does not have this cold start issue. it is extremely greedy for early learning. 
                (lowers performance ceiling but whatever)
        -way more sensetive to hyperparameters than the DQN side.
            -if you choose a LR too big, the policy diverges and you get NANs or maxed out mus/sigmas
            -if Tau is too low you get "Value Runaways" like in vanilla DQN:
                >>>
                    total samples: 53735, ep 204: high-score     1600.000, score      941.000
                    total samples: 54080, ep 205: high-score     1600.000, score      345.000
                    total samples: 54349, ep 206: high-score     1600.000, score      269.000
                    total samples: 54856, ep 207: high-score     1600.000, score      507.000
                    total samples: 54865, ep 208: high-score     1600.000, score        9.000
                    total samples: 54874, ep 209: high-score     1600.000, score        9.000
                    total samples: 54882, ep 210: high-score     1600.000, score        8.000
        -can semi-silently fail if youre action ranges are not bounded correctly. (lol)

        Potential:
        -not satisfactorally stable for a drop in use in continuous envs, but 
            with twinning, td clamping, etc, the policy stability should clean and then we 
            basically end up with PPO

        Output Sample:
        >>>
            total samples: 4769, ep 56: high-score      350.000, score      316.000
            total samples: 4992, ep 57: high-score      350.000, score      223.000
            total samples: 5219, ep 58: high-score      350.000, score      227.000
            total samples: 5435, ep 59: high-score      350.000, score      216.000
            total samples: 5631, ep 60: high-score      350.000, score      196.000
            ...
            total samples: 9823, ep 74: high-score      490.000, score      192.000
            total samples: 9988, ep 75: high-score      490.000, score      165.000
            total samples: 10776, ep 76: high-score      788.000, score      788.000
            ...

            total samples: 21462, ep 115: high-score      788.000, score      429.000
            total samples: 21820, ep 116: high-score      788.000, score      358.000
            total samples: 22145, ep 117: high-score      788.000, score      325.000
            total samples: 22530, ep 118: high-score      788.000, score      385.000
            total samples: 22914, ep 119: high-score      788.000, score      384.000
            total samples: 23155, ep 120: high-score      788.000, score      241.000
            total samples: 23680, ep 121: high-score      788.000, score      525.000
'''

class ReplayBuffer:
    def __init__(self, size, state_shape, num_actions):
        self.size = size
        self.count = 0

        self.state_memory       = np.zeros((self.size, *state_shape), dtype=np.float32)
        self.action_memory      = np.zeros((self.size, num_actions),  dtype=np.float32)
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

class Critic(torch.nn.Module):
    def __init__(self, learn_rate, state_shape, num_actions, device, layers):
        super().__init__()
        self.fc1_dims = layers[0]
        self.fc2_dims = layers[1]
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.device = device

        self.state_encoder  = nn.Linear(*self.state_shape, self.fc1_dims)
        self.action_encoder = nn.Linear( self.num_actions, self.fc1_dims)
        self.fc2 = nn.Linear( self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear( self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learn_rate)
        self.to(self.device)

    def forward(self, states, actions):
        states_encoded = self.state_encoder(states)
        actions_encoded = self.action_encoder(actions)
        states_actions = F.relu(torch.add(states_encoded, actions_encoded))

        x = F.relu(self.fc2(states_actions))
        values = self.fc3(x)

        return values

class Actor(torch.nn.Module):
    def __init__(self, learn_rate, state_shape, num_actions, device, layers):
        super().__init__()
        self.fc1_dims = layers[0]
        self.fc2_dims = layers[1]
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.device = device

        self.fc1 = nn.Linear(*self.state_shape, self.fc1_dims)
        self.fc2 = nn.Linear( self.fc1_dims,    self.fc2_dims)
        self.fc3 = nn.Linear( self.fc2_dims,    self.num_actions * 2)

        self.optimizer = optim.Adam(self.parameters(), lr=learn_rate)
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        policy = x.view(-1, self.num_actions, 2)

        return policy

class Agent():
    def __init__(self, learn_rate, state_shape, num_actions, batch_size, layers):
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu,")
        self.device = torch.device("cpu")

        self.actor = Actor(learn_rate, state_shape, num_actions, self.device, layers)
        self.target_actor = Actor(learn_rate, state_shape, num_actions, self.device, layers)
        self.critic = Critic(learn_rate, state_shape, num_actions, self.device, layers)
        self.target_critic = Critic(learn_rate, state_shape, num_actions, self.device, layers)

        self.memory = ReplayBuffer(size=100000, state_shape=state_shape, num_actions=num_actions)
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.gamma = 0.99
        self.tau = 0.2

    def choose_action(self, states, target=False, batch=True):
        if not batch:
            states = torch.tensor(states).float().detach()
            states = states.to(self.device)
            states = states.unsqueeze(0)

        if not target:
            self.actor.eval()
            policy = self.actor(states)
        else:   #   use target network instead
            self.target_actor.eval()
            policy = self.target_actor(states)

        mus, sigmas = policy[:, :, 0], policy[:, :, 1]
        sigmas = torch.exp(sigmas)

        actions_dist = torch.distributions.Normal(mus, sigmas)

        if not batch:
            # print(f"{mus} {sigmas}")
            actions = actions_dist.sample()
            actions = torch.tanh(actions)
            actions = actions[0]
            actions = actions.detach().cpu().numpy()
        else:
            actions = actions_dist.rsample()    #   not sure if should be rsample
            actions = torch.tanh(actions)

        return actions

    def store_memory(self, state, actions, reward, state_, done):
        self.memory.store_memory(state, actions, reward, state_, done)

    def update_params(self):
        actor_params = self.actor.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        actor_dict = dict(actor_params)
        target_actor_dict = dict(target_actor_params)
        critic_dict = dict(critic_params)
        target_critic_dict = dict(target_critic_params)

        for name in actor_dict:
            actor_dict[name] = (
                self.tau * actor_dict[name].clone()
                + (1 - self.tau) * target_actor_dict[name].clone())
        self.target_actor.load_state_dict(actor_dict)

        for name in critic_dict:
            critic_dict[name] = (
                self.tau * critic_dict[name].clone()
                + (1 - self.tau) * target_critic_dict[name].clone())
        self.target_critic.load_state_dict(critic_dict)

    def learn(self):
        if self.memory.count < self.batch_size:
            return

        states, actions, rewards, states_, dones = \
            self.memory.sample(self.batch_size)
        states  = torch.tensor( states  ).to(self.device)
        actions = torch.tensor( actions ).to(self.device)
        rewards = torch.tensor( rewards ).to(self.device)
        states_ = torch.tensor( states_ ).to(self.device)
        dones   = torch.tensor( dones   ).to(self.device)
        
        self.target_critic.eval()
        self.target_actor.eval()
        self.critic.eval()

        values = self.critic(states, actions)

        actions_ = self.choose_action(states_, target=True)
        values_ = self.target_critic(states_, actions_)
        values_[dones] = 0.0

        target = rewards + self.gamma * values_ 
        td = target - values       

        self.critic.train()
        critic_loss = (td**2).mean()
        
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        ''' update based on new policy of old states '''
        self.critic.eval()
        retrospective_actions = self.choose_action(states, target=False)
        self.actor.train()
        retrospective_values = self.critic(states, retrospective_actions)
        actor_loss = torch.mean(-retrospective_values)
        
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_params()

if __name__ == '__main__':
    env = ContinuousCartPoleEnv()
    agent = Agent(learn_rate=0.001, state_shape=(4,), num_actions=1, batch_size=64, layers=(256, 128))

    high_score = -math.inf
    episode = 0
    num_samples = 0
    while True:
        done = False
        state = env.reset()

        score, frame = 0, 1
        while not done:
            env.render()

            actions = agent.choose_action(state, batch=False)
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