import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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