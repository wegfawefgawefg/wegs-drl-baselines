from model import Network
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from cpprb import ReplayBuffer

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
    def __init__(self, lr, state_shape, num_actions, batch_size, 
            max_mem_size=1000):
        self.lr = lr
        self.gamma = 0.99
        self.action_space = list(range(num_actions))
        self.batch_size = batch_size
        self.target_update_interval = 200
        self.step_count = 0

        self.epsilon = Lerper(start=1.0, end=0.01, num_steps=2000)

        self.memory = ReplayBuffer(
            max_mem_size, 
            {   "obs":      { "shape": state_shape  },
                "act":      { "shape": 1            },
                "rew":      {                       },
                "next_obs": { "shape": state_shape  },
                "done":     { "shape": 1            }})

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")

        self.V_MIN, self.V_MAX = 0, 200
        self.NUM_ATOMS = 4
        self.support = torch.linspace(
            self.V_MIN, self.V_MAX, self.NUM_ATOMS).to(self.device)
        self.net = Network(lr, state_shape, num_actions, 
            self.support, self.NUM_ATOMS).to(self.device)
        self.net_ = Network(lr, state_shape, num_actions, 
            self.support, self.NUM_ATOMS).to(self.device)

        self.net_.load_state_dict(self.net.state_dict())

    def choose_action(self, observation):
        if np.random.random() > self.epsilon.value():
            state = torch.tensor(observation).float().detach()
            state = state.to(self.device)
            state = state.unsqueeze(0)

            q_values = self.net(state)
            action = torch.argmax(q_values).item()
            return action
        else:
            return np.random.choice(self.action_space)

    def store_memory(self, state, action, reward, next_state, done):
        self.memory.add(obs=state, act=action, rew=reward, next_obs=next_state, done=done)  

    def learn(self):
        if self.memory.get_stored_size() < self.batch_size:
            return
    
        batch = self.memory.sample(self.batch_size)
            
        states  = torch.tensor( batch["obs"]                        ).to(self.device)
        actions = torch.tensor( batch["act"],   dtype=torch.int64   ).to(self.device).T[0]
        rewards = torch.tensor( batch["rew"]                        ).to(self.device)
        states_ = torch.tensor( batch["next_obs"]                   ).to(self.device)
        dones   = torch.tensor( batch["done"],  dtype=torch.float32 ).to(self.device)

        batch_index = np.arange(self.batch_size, dtype=np.int64)

        #   the difference between each reward quanta
        delta_z = float(self.V_MAX - self.V_MIN) / (self.NUM_ATOMS - 1)    #28.571428571428573

        with torch.no_grad():
            qs_ = self.net_(states_)         #[64,2]
            actions_ = qs_.argmax(dim=1)    #[64]
            dists_ = self.net_.dist(states_) #[64,2,8]
            action_dist_ = dists_[batch_index, actions_] #[64,8]

            # print(action_dist_)
            # print(action_dist_.shape)
            # quit()

            #done    #[64,1]
            #reward  #[64,1]
            #support #[51]
            print("support")
            print(self.support)
            print(self.support.shape)
            t_z = rewards + (1-dones) * self.gamma * self.support #   shape=[64,8]
            # t_z = torch.tensor((self.batch_size,)).to(self.device) * self.support
            t_z = torch.zeros((self.batch_size, self.NUM_ATOMS)).to(self.device)
            tzindxs = np.arange(6)
            t_z[tzindxs] = self.support

            print("t-z")
            print(t_z)
            print(t_z.shape)
            # quit()

            #   normalization bullshit
            t_z = t_z.clamp(min=self.V_MIN, max=self.V_MAX)
            b = (t_z - self.V_MIN) / delta_z    #   quantize
            l = b.floor().long()                #   indices
            u = b.ceil().long()                 #   offsets to the closest reward bracket

            print(t_z)
            print(t_z.shape)
            # quit()

            print(b)
            print(b.shape)
            print(l)
            print(l.shape)
            print(u)
            print(u.shape)
            # quit()

            #   this is a giant indexing array
            offset = (  #[64,8] #[[0..0],[8..8],[16..16],,,[504..504]
                torch.linspace(
                    0, (self.batch_size - 1) * self.NUM_ATOMS, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.NUM_ATOMS)
                .to(self.device)
            )

            print("\noffset")
            print(offset)
            print(offset.shape)

            frac = u.float() - b        #   percentages, decreasing, axis = 1
            dec_frac = b - l.float()    #   percentages, increasing, axis = 1

            # print(something_else)
            # print(something_else.shape)
            # quit()

            action_dist_ = torch.ones((self.batch_size, self.NUM_ATOMS)).to(self.device)

            proj_dist = torch.zeros(action_dist_.size(), device=self.device)   #   [64,8]

            print("proj_dist")
            print(proj_dist)
            print(proj_dist.shape)

            print("action_dist_")
            print(action_dist_)
            print(action_dist_.shape)
            # print(frac)
            # print(frac.shape)

            print("l")
            print(l)
            print(l.shape)

            print("offset")
            print(offset)
            print(offset.shape)

            proj_dist.view(-1).index_add_(  #[64,8]
                0, 
                (l + offset).view(-1), 
                (action_dist_).view(-1)#(action_dist_ * frac).view(-1)
            )
            print("RESULT: proj_dist")
            print(proj_dist)
            print(proj_dist.shape)
            proj_dist.view(-1).index_add_(  #[64,8]
                0, 
                (u + offset).view(-1), 
                (action_dist_).view(-1)#(action_dist_ * dec_frac).view(-1)
            )

            print("proj_dist")
            print(proj_dist)
            print(proj_dist.shape)
            quit()


            # print(dec_frac)
            # print(dec_frac.shape)
            # quit()

        # print(actions)
        # print(actions.shape)
        # quit()

        dists = self.net.dist(states) #[64,2,8]
        log_p = torch.log(dists[batch_index, actions])

        loss = -(proj_dist * log_p).sum(1).mean()

        self.net.optimizer.zero_grad()
        loss.backward()
        self.net.optimizer.step()

        self.epsilon.step()

        self.step_count += 1

        if self.step_count % self.target_update_interval == 0:
            print("targnet update!!")
            self.net_.load_state_dict(self.net.state_dict())

        return loss