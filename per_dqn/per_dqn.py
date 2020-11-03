import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cpprb import PrioritizedReplayBuffer

class Network(torch.nn.Module):
    def __init__(self, learn_rate, inputShape, numActions):
        super().__init__()
        self.inputShape = inputShape
        self.numActions = numActions
        self.fc1Dims = 1024
        self.fc2Dims = 512

        # print("input shape {}".format(self.inputShape))
        self.fc1 = nn.Linear(*self.inputShape, self.fc1Dims)
        self.fc2 = nn.Linear(self.fc1Dims, self.fc2Dims)
        self.fc3 = nn.Linear(self.fc2Dims, numActions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

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
            max_mem_size=100000):
        self.lr = lr
        self.gamma = 0.99
        self.action_space = list(range(num_actions))
        self.batch_size = batch_size

        self.epsilon = Lerper(start=1.0, end=0.01, num_steps=2000)
        self.importance_exp = Lerper(start=0.4, end=1.0, num_steps=100000)

        self.priority_exp = 0.6
        self.memory = PrioritizedReplayBuffer(
            max_mem_size, 
            {   "obs":      { "shape": state_shape  },
                "act":      { "shape": 1            },
                "rew":      {                       },
                "next_obs": { "shape": state_shape  },
                "done":     { "shape": 1            }
            },
            alpha = self.priority_exp)

        self.net = Network(lr, state_shape, num_actions)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon.value():
            state = torch.tensor(observation).float().detach()
            state = state.to(self.net.device)
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
    
        batch = self.memory.sample(self.batch_size, self.importance_exp.value())
            
        states  = torch.tensor( batch["obs"]                     ).to(self.net.device)
        actions = torch.tensor( batch["act"],   dtype=torch.int64).to(self.net.device).T[0]
        rewards = torch.tensor( batch["rew"]                     ).to(self.net.device).T[0]
        states_ = torch.tensor( batch["next_obs"]                ).to(self.net.device)
        dones   = torch.tensor( batch["done"],  dtype=torch.bool ).to(self.net.device).T[0]
        weights = torch.tensor( batch["weights"]                 ).to(self.net.device)

        batch_index = np.arange(self.batch_size, dtype=np.int64)

        q_values  =   self.net(states)[batch_index, actions]
        q_values_ =   self.net(states_)

        action_qs_ = torch.max(q_values_, dim=1)[0]
        action_qs_[dones] = 0.0
        q_target = rewards + self.gamma * action_qs_

        td = q_target - q_values

        self.net.optimizer.zero_grad()
        loss = ((td ** 2.0) * weights).mean()
        loss.backward()
        self.net.optimizer.step()

        new_priorities = (td.abs()).detach().cpu()
        self.memory.update_priorities(batch["indexes"], new_priorities)

        self.epsilon.step()
        self.importance_exp.step()

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