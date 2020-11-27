import math
import random

import gym       
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

'''
        --  Simpleton's Deep Deterministic Policy Gradient --
        DDPG is essentially Batch Soft-Double Actor Critic,
        but with some noise added in for exploration. In the continous 
        domain its possible the noise actually helps. 
        However, in the discrete domain, since the noise is 
        not learnable it's probably not much better than a fairly selected epsilon greedy.

        I'm not going to include the noise. And I will use a normal Q-Function.
        Additionally I dont care about the deterministic component.
        Basically this is shitty batch actor critic.

        Goal:
        -to bring q(s, a) to continuous action spaces
        -to try and make the policy more stable by aiming at a nonstochastic 
            actor??
        
        Performance:
        -better than dqn in atari (k)
            
        Flaws:
        -complicated to implement
        -unclear what is the source of the good performance of ddpg, 
            the fundamental idea or the other random shit in the paper.
                (noise)
                (batch norm)

        Potential:
        -hopefully none. I hope ddpg is outmoded.

        Output Sample:
        >>>
            total samples: 22260, ep 203: high-score      521.000, score      240.000
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

class Critic(torch.nn.Module):
    def __init__(self, learn_rate, state_shape, num_actions, device):
        super().__init__()
        self.fc1_dims = 1024
        self.fc2_dims = 512
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
        batch_size = actions.shape[0]

        actions = actions.long().detach()
        action_one_hots = torch.zeros(
            (batch_size, self.num_actions), dtype=torch.float32).to(self.device)
        indices = torch.arange(batch_size, dtype=torch.int64).to(self.device)
        action_one_hots[indices, actions] = 1.0

        states_encoded = self.state_encoder(states)
        actions_encoded = self.action_encoder(action_one_hots)
        states_actions = F.relu(torch.add(states_encoded, actions_encoded))

        x = F.relu(self.fc2(states_actions))
        values = self.fc3(x)

        return values

class Actor(torch.nn.Module):
    def __init__(self, learn_rate, state_shape, num_actions, device):
        super().__init__()
        self.fc1_dims = 1024
        self.fc2_dims = 512
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.device = device

        self.fc1 = nn.Linear(*self.state_shape, self.fc1_dims)
        self.fc2 = nn.Linear( self.fc1_dims,    self.fc2_dims)
        self.fc3 = nn.Linear( self.fc2_dims,    self.num_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=learn_rate)
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy = self.fc3(x)

        return policy

class Agent():
    def __init__(self, learn_rate, state_shape, num_actions, batch_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")

        self.actor = Actor(learn_rate, state_shape, num_actions, self.device)
        self.target_actor = Actor(learn_rate, state_shape, num_actions, self.device)
        self.critic = Critic(learn_rate, state_shape, num_actions, self.device)
        self.target_critic = Critic(learn_rate, state_shape, num_actions, self.device)

        self.memory = ReplayBuffer(size=100000, state_shape=state_shape)
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.gamma = 0.99
        self.tau = 0.1

    def choose_action(self, observation):
        state = torch.tensor(observation).float().detach()
        state = state.to(self.device)
        state = state.unsqueeze(0)
        
        self.actor.eval()
        policy = self.actor(state)

        policy = policy[0]
        policy = F.softmax(policy, dim=0)
        actions_probs = torch.distributions.Categorical(policy)
        action = actions_probs.sample()
        return action.item()

    def choose_actions(self, states, target=False):
        if not target:
            self.actor.eval()
            policy = self.actor(states)
        else:   #   use target network instead
            self.target_actor.eval()
            policy = self.target_actor(states)

        policy = F.softmax(policy, dim=1)
        actions_probs = torch.distributions.Categorical(policy)
        actions = actions_probs.sample()

        return actions

    def store_memory(self, state, action, reward, state_, done):
        self.memory.store_memory(state, action, reward, state_, done)

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

        actions_ = self.choose_actions(states_, target=True)
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
        retrospective_actions = self.choose_actions(states, target=False)
        self.actor.train()
        retrospective_values = self.critic(states, retrospective_actions)
        actor_loss = torch.mean(-retrospective_values)
        
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_params()

if __name__ == '__main__':
    env = gym.make('CartPole-v1').unwrapped
    agent = Agent(learn_rate=0.001, state_shape=(4,), num_actions=2, batch_size=64)

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