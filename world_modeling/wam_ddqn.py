import math
import random

import gym       
from cartpole_visualizer import SettableCartPoleEnv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

'''
        --  Training A World Action Model --
        Give the world model the current action as well.

        Goal:
        -make the world model not have to predict the action taken as well

        Performance:
        -should be identical to ddqn still
        -the world model doesnt seem any more precise... very similar loss

        Flaws:
        -predicted next state is noisy, 
        -has a hard time predicting next state for uncommon states (noise increases. qualitatively obvious)

        Potential:
        -this is near intrinsic curiosity module
        -may help performance more in more interesting environment

        Output Sample:
        >>>
            wm_error 0.5000302089496248
            wm_error 0.48847666408228785
            wm_error 0.49641973881662504
            wm_error 0.4900784474355991
            wm_error 0.4965915794698861
            wm_error 0.49249339059316666
            wm_error 0.49592514688093187
            wm_error 0.48827728794249037

    Coder Notes:   
        -rearanged the training code a little bit, 
        -had to do some fuckery to concatenate the onehot encodings of the actions to the states
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
        self.fc1Dims = 128
        self.fc2Dims = 64

        self.fc1 = nn.Linear(*input_shape,  self.fc1Dims)
        self.fc2 = nn.Linear( self.fc1Dims, self.fc2Dims)
        self.fc3 = nn.Linear( self.fc2Dims, num_actions  )

        self.optimizer = optim.Adam(self.parameters(), lr=learn_rate)
        self.loss = nn.MSELoss()
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class WorldModel(torch.nn.Module):
    ''' predicts next_state from state  '''
    def __init__(self, learn_rate, input_shape, state_shape):
        super().__init__()
        self.fc1Dims = 64
        self.fc2Dims = 64

        self.fc1 = nn.Linear(*input_shape,  self.fc1Dims)
        self.fc2 = nn.Linear( self.fc1Dims, self.fc2Dims)
        self.fc3 = nn.Linear( self.fc2Dims, *state_shape)

        self.optimizer = optim.Adam(self.parameters(), lr=learn_rate)
        self.loss = nn.MSELoss()
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class LinearSchedule:
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

class Agent():
    def __init__(self, learn_rate, input_shape, num_actions, batch_size):
        self.net = Network(learn_rate, input_shape, num_actions)
        self.target_net = Network(learn_rate, input_shape, num_actions)
        self.world_model = WorldModel(learn_rate, 
            input_shape=(input_shape[0] + num_actions, ),
            state_shape=input_shape)

        self.memory = ReplayBuffer(size=100000, state_shape=input_shape)
        self.epsilon = LinearSchedule(start=1.0, end=0.01, num_steps=2000)

        self.batch_size = batch_size
        self.num_actions = num_actions
        self.gamma = 0.99

        self.learn_step_counter = 0
        self.net_copy_interval = 10

    def choose_action(self, observation):
        if random.random() > self.epsilon.value():
            state = torch.tensor(observation).float().detach()
            state = state.to(self.net.device)
            state = state.unsqueeze(0)

            q_values = self.net(state)
            action = torch.argmax(q_values).item()

            return action
        else:
            action = random.randint(0, self.num_actions - 1)
            return action

    def store_memory(self, state, action, reward, state_, done):
        self.memory.store_memory(state, action, reward, state_, done)

    def predict_next_state(self, state, action):
        self.world_model.eval()

        state = torch.tensor(state).float().detach()
        state = state.to(self.net.device)

        action = torch.tensor(action).long().detach()
        action = action.to(self.net.device)
        action = action.unsqueeze(0)

        action_one_hot = torch.zeros(self.num_actions)
        action_one_hot[action] = 1.0

        state_action = torch.cat([state, action_one_hot])
        state_ = self.world_model(state_action)
        state_ = state_.cpu().detach().numpy()

        return state_

    def predict_next_states(self, states, actions):
        #   one hot encode the actions
        actions = actions.long().detach()
        action_one_hots = torch.zeros((self.batch_size, self.num_actions), dtype=torch.float32)
        indices = torch.arange(self.batch_size, dtype=torch.int64)
        action_one_hots[indices, actions] = 1.0

        states_actions = torch.cat([states, action_one_hots], dim=1)
        states_ = self.world_model(states_actions)

        return states_

    def train_world_model(self, states, actions):
        self.world_model.train()
        
        p_states_ = self.predict_next_states(states, actions)
        world_model_loss = self.world_model.loss(p_states_, states)

        self.world_model.optimizer.zero_grad()
        world_model_loss.backward()
        self.world_model.optimizer.step()

    def train_agent(self, states, actions, rewards, states_, dones):
        self.net.train()

        batch_indices = np.arange(self.batch_size, dtype=np.int64)
        action_qs = self.net(states)[batch_indices, actions]    #   (batch_size, 1)

        qs_ =   self.target_net(states_)            #   (batch_size, num_actions)
        policy_qs = self.net(states_)               #   (batch_size, num_actions)
        actions_ = torch.max(policy_qs, dim=1)[1]   #   (batch_size, 1)
        action_qs_ = qs_[batch_indices, actions_]
        action_qs_[dones] = 0.0

        q_targets = rewards + self.gamma * action_qs_

        loss = self.net.loss(q_targets, action_qs).to(self.net.device)

        self.net.optimizer.zero_grad()
        loss.backward()
        self.net.optimizer.step()

    def learn(self):
        if self.memory.count < self.batch_size:
            return
    
        states, actions, rewards, states_, dones = \
            self.memory.sample(self.batch_size)
        states  = torch.tensor( states  ).to(self.net.device)
        actions = torch.tensor( actions ).to(self.net.device)
        rewards = torch.tensor( rewards ).to(self.net.device)
        states_ = torch.tensor( states_ ).to(self.net.device)
        dones   = torch.tensor( dones   ).to(self.net.device)

        self.train_world_model(states, actions)
        self.train_agent(states, actions, rewards, states_, dones)

        self.epsilon.step()
        if self.learn_step_counter % self.net_copy_interval == 0:
            self.target_net.load_state_dict(self.net.state_dict())
        self.learn_step_counter += 1

if __name__ == '__main__':
    env = gym.make('CartPole-v1').unwrapped
    visualizer_env = SettableCartPoleEnv()
    agent = Agent(learn_rate=0.001, input_shape=(4,), num_actions=2, batch_size=64)

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

            predicted_next_state = agent.predict_next_state(state, action)
            visualizer_env.set_state(predicted_next_state)
            visualizer_env.render()

            world_model_error = np.abs(predicted_next_state - state_).sum()
            print(f"wm_error {world_model_error}")

            state = state_

            num_samples += 1
            score += reward
            frame += 1

        high_score = max(high_score, score)

        print(("total samples: {}, ep {}: high-score {:12.3f}, score {:12.3f}").format(
            num_samples, episode, high_score, score, frame))

        episode += 1