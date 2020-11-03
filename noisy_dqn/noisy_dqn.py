import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cpprb import ReplayBuffer

class NoisyLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))

        #   prevents bias_epsilon from being learned on, 
        #   but lets it remain in the state dict as a param
        self.register_buffer("weight_epsilon", torch.Tensor(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        ''' more inputs:
                the tighter the noise start, 
                the tighter the std deviation
            less inputs:
                much wider noise midpoint range
                looser noise curve

            why:
                noise should be more subtle with more inpts?
        '''
        mu_range = 1 / math.sqrt(self.in_features)
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features))

    @staticmethod
    def scale_noise(size):
        '''This is factorized gaussian noise. 
            It seems the gradients will flow back to this single tensor which 
            only has one noise variable per in/out features, as opposed to 
            one for each weight and bias. This tensor is converted to a diagonal 
            matrix with the outer product. and that diagonal is then applied 
            to each weight individually. Surely there are agent performance 
            implications, but using factorized gaussian as opposed to 
            independant gaussian noise reduces the number of params to learn by about 
            sqrt(m*n). Thats why we sqrt the values. It is multiplied by 
            itself in the outer product (process of making the diag matrix). 
            So to keep the values from getting big, you sqrt them.
            '''
        x = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=size))
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,)

class Network(torch.nn.Module):
    def __init__(self, alpha, input_shape, num_actions):
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.fc1_dims = 1024
        self.fc2_dims = 512

        self.fc1 = nn.Linear(  *self.input_shape, self.fc1_dims)
        self.fc2 = NoisyLinear( self.fc1_dims,    self.fc2_dims)
        self.fc3 = NoisyLinear( self.fc2_dims,    num_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def reset_noise(self):
        self.fc2.reset_noise()
        self.fc3.reset_noise()

class Lerper:
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

class Agent:
    def __init__(self, learn_rate, state_shape, num_actions, batch_size):
        self.mem_size=100000
        self.gamma = 0.99
        self.action_space = list(range(num_actions))
        self.batch_size = batch_size

        self.epsilon = Lerper(start=1.0, end=0.01, num_steps=2000)

        self.memory = ReplayBuffer(
            self.mem_size, 
            {   "obs":      { "shape": state_shape  },
                "act":      { "shape": 1            },
                "rew":      {                       },
                "next_obs": { "shape": state_shape  },
                "done":     { "shape": 1            }})

        self.net = Network(learn_rate, state_shape, num_actions)

    def choose_action(self, observation):
        state = torch.tensor(observation).float().detach()
        state = state.to(self.net.device)
        state = state.unsqueeze(0)

        q_values = self.net(state)
        action = torch.argmax(q_values).item()
        return action

    def store_memory(self, state, action, reward, next_state, done):
        self.memory.add(obs=state, act=action, rew=reward, next_obs=next_state, done=done)  

    def learn(self):
        if self.memory.get_stored_size() < self.batch_size:
            return
    
        batch = self.memory.sample(self.batch_size)
            
        states  = torch.tensor( batch["obs"]                     ).to(self.net.device)
        actions = torch.tensor( batch["act"],   dtype=torch.int64).to(self.net.device).T[0]
        rewards = torch.tensor( batch["rew"]                     ).to(self.net.device).T[0]
        states_ = torch.tensor( batch["next_obs"]                ).to(self.net.device)
        dones   = torch.tensor( batch["done"],  dtype=torch.bool ).to(self.net.device).T[0]

        batch_index = np.arange(self.batch_size, dtype=np.int64)

        q_values  =   self.net(states)[batch_index, actions]
        q_values_ =   self.net(states_)

        action_qs_ = torch.max(q_values_, dim=1)[0]
        action_qs_[dones] = 0.0
        q_target = rewards + self.gamma * action_qs_

        td = q_target - q_values

        self.net.optimizer.zero_grad()
        loss = (td ** 2.0).mean()
        loss.backward()
        self.net.optimizer.step()

        self.net.reset_noise()

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
            if frame == 5000:
                break
            env.render()

            action = agent.choose_action(state)
            state_, reward, done, info = env.step(action)
            agent.store_memory(state, action, reward, state_, done)
            agent.learn()

            state = state_

            score += reward
            frame += 1
            num_samples += 1

        high_score = max(high_score, score)

        print(( "num-samples: {}, ep {}: high-score {:12.3f}, "
                "score {:12.3f}, epsilon {:6.3f}").format(
            num_samples, episode, high_score, score, agent.epsilon.value()))
        episode += 1