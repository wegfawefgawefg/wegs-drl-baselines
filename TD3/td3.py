import math
import random
import copy 
import itertools

import gym       
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from continuous_cartpole import ContinuousCartPoleEnv

'''
        --  Twin Delayed Deeply Deterministic Policy Gradient --

        DDPG has value error just like DQN, but target networks are apparently not enough to 
        prevent runaway state-values. TD3 tries to fix DDPG's runaway state-values.

        TD3 adds 3 things:
        1. Clip the state-value estimates
        2. Policy updates should be slower than Value updates, apparently... 
            (This contradicts what they said in the paper about DQN's fast policy changes 
             preventing runaway state-values, however the paper demonstrated that the moving 
             policy is responsible for a few orders of magnitude worth of variance in state-value
             so I'd have to say this seems reasonable. In the real world though you can never expect 
             accurate value predictions. The environment is just too rich. So maybe this is mostly useful 
             for toy environments. I guess you would want to update the policy as slow as possible, 
             without becoming so slow as to make the agent not lose out to the competition.)
            -add delays, and slower policy learn rates
        3. Adding fucking action layer noise back... (basically epsilon greedy) 
            (This contradicts the original premise of DDPG in the first place.)

        Goal:
        -fix runaway state-values in ddpg
        -unintentional: lower policy change rate; raises upper performance bounds in late stages of training

        Performance:

        Flaws:

        Potential:
        -I know the paper shows such crazy good scores, but people seem to be getting wildly different results 
         with different baseline tunings. So it's a repeatability problem that makes you wonder if it's actually 
         better. (It probably is. In the paper they added these stability modifications to DQN-AC and got 
         some great scores.)

        Output Sample:
        >>>
            total samples: 4769, ep 56: high-score      350.000, score      316.000
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

class Critic(torch.nn.Module):
    def __init__(self, layer_sizes, state_shape, num_actions):
        super().__init__()
        self.fc1_dims = layer_sizes[0]
        self.fc2_dims = layer_sizes[1]

        self.state_encoder  = nn.Linear(*state_shape, self.fc1_dims)
        self.action_encoder = nn.Linear( num_actions, self.fc1_dims)
        self.fc2 = nn.Linear( self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear( self.fc2_dims, 1)

    def forward(self, states, actions):
        states_encoded = self.state_encoder(states)
        actions_encoded = self.action_encoder(actions)
        states_actions = F.relu( torch.add(states_encoded, actions_encoded) )

        x = F.relu(self.fc2(states_actions))
        values = self.fc3(x)
        
        return values

class Actor(torch.nn.Module):
    def __init__(self, layer_sizes, state_shape, num_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(*state_shape,    layer_sizes[0]),     nn.ReLU(),
            nn.Linear( layer_sizes[0], layer_sizes[1]),     nn.ReLU(),
            nn.Linear( layer_sizes[1], num_actions))

    def forward(self, x):
        x = self.model(x)
        actions = torch.tanh(x)
        # print(x)
        # print(actions)
        return actions
        # return x   #    tanh or not

class Agent():
    def __init__(self, 
            #   basics
            state_shape, 
            num_actions, 
            action_range,
            layer_sizes=(256, 256), 
            batch_size=8,#64,
            gamma=0.99,

            #   replay buffer
            buffer_size=1_000_000,      
            min_buffer_fullness=1_000, #10_000,

            #   learn rates
            critic_learn_rate=0.001,    
            actor_learn_rate=0.001,     
            target_lerp_rate=0.995,

            #   update frequencies / delays
            num_babbling_steps=1_000, #10_000,
            update_every=50,#50,
            value_maturity_delay=100, #1_000,
            policy_delay=2,

            #   noise
            actor_noise=0.1,
            target_noise=0.2,
            noise_clamp=0.5
            ):

        '''   SETTINGS   '''
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu,")
        self.DEVICE               = torch.device("cpu")
 
        self.NUM_ACTIONS          = num_actions
        self.ACTION_RANGE         = action_range
        self.BATCH_SIZE           = batch_size
        self.GAMMA                = gamma
        self.MIN_BUFFER_FULLNESS  = min_buffer_fullness
        self.NUM_BABBLING_STEPS   = num_babbling_steps
        self.UPDATE_EVERY         = update_every
        self.VALUE_MATURITY_DELAY = value_maturity_delay
        self.POLICY_DELAY         = policy_delay
        self.TAU                  = target_lerp_rate
        self.TARGET_NOISE_MAG     = target_noise
        self.ACTOR_NOISE_MAG      = actor_noise
        self.NOISE_CLAMP          = 0.5

        '''   STATE   '''
        self.learn_count = 0
        self.critic_updates_count = 0
        self.babbling_steps = 0
        self.memory = ReplayBuffer(buffer_size, state_shape, num_actions)

        self.actor      = Actor( layer_sizes, state_shape, num_actions)
        self.critic_one = Critic(layer_sizes, state_shape, num_actions)
        self.critic_two = Critic(layer_sizes, state_shape, num_actions)

        self.target_actor      = copy.deepcopy(self.actor)
        self.target_critic_one = copy.deepcopy(self.critic_one)
        self.target_critic_two = copy.deepcopy(self.critic_two)
        
        self.critic_params = itertools.chain(
            self.critic_one.parameters(), 
            self.critic_two.parameters())
        self.target_critic_params = itertools.chain(
            self.target_critic_one.parameters(), 
            self.target_critic_two.parameters())

        ''' freeze all target network weights'''
        for param in self.target_actor.parameters():
            param.requires_grad = False
        for param in self.target_critic_params:
            param.requires_grad = False

        self.actor_optimizer = optim.Adam(  params=self.actor.parameters(), lr=actor_learn_rate)
        self.critic_optimizer = optim.Adam( params=self.critic_params,      lr=critic_learn_rate)

    def store_memory(self, state, actions, reward, state_, done):
        self.memory.store_memory(state, actions, reward, state_, done)

    def update_actor_target_params(self):
        ''' time this operation to find if "in-place" is shorter method ''' 
        # with torch.no_grad():
        #     for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
        #         target_param.data.mul_(self.TAU)
        #         target_param.data.add_((1 - self.TAU) * param.data)

        with torch.no_grad():
            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

    def update_critic_target_params(self):
        with torch.no_grad():
            for param, target_param in zip(self.critic_params, self.target_critic_params):
                target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

    def update_all_target_params(self):
        self.update_critic_target_params()
        self.update_actor_target_params()

    def choose_actions(self, states, noisy=False, noise_mag=0.0, target=False):
        if target:
            actions = self.target_actor(states)
        else:
            actions = self.actor(states)
        
        actions = self.ACTION_RANGE * actions
        # print(f"actions: {actions}")
        if noisy:
            epsilon = torch.randn_like(actions) * noise_mag
            noise = torch.clamp(epsilon, -self.NOISE_CLAMP, self.NOISE_CLAMP)
            # print(f"action: {actions}, noise: {noise}")
            actions += noise
        actions = torch.clamp(actions, -self.ACTION_RANGE, self.ACTION_RANGE)

        return actions

    def choose_action(self, states, noisy=True):
        ''' this is the non learnable version of choose_actions() 
            for taking single actions in the env                    '''
        if self.babbling_steps <= self.NUM_BABBLING_STEPS:
            print(("Babbling Mode: ignoring policy. choosing random action. "
                f"(steps remaining: {self.NUM_BABBLING_STEPS - self.babbling_steps})"))
            actions = np.random.uniform(low=-self.ACTION_RANGE, high=self.ACTION_RANGE, size=self.NUM_ACTIONS)
            self.babbling_steps += 1
        else:
            with torch.no_grad():
                states = torch.tensor(states).float().detach()
                states = states.to(self.DEVICE)
                states = states.unsqueeze(0)

                actions = self.choose_actions(states, noisy, noise_mag=self.ACTOR_NOISE_MAG, target=False)

                actions = actions[0]
                actions = actions.detach().cpu().numpy()

        return actions

    def learn(self):
        if self.babbling_steps <= self.NUM_BABBLING_STEPS:
            return
        elif (self.memory.count < self.MIN_BUFFER_FULLNESS):
            print(("Delaying Learning: minumum buffer fullness not achieved. " 
                f"(remaining needed memories: {self.MIN_BUFFER_FULLNESS - self.memory.count})"))
            return
        elif self.learn_count % self.UPDATE_EVERY == 0:
            for _ in range(self.UPDATE_EVERY):
                states, actions, rewards, states_, dones = self.memory.sample(self.BATCH_SIZE, self.DEVICE)
                
                '''                     critic losses                       '''
                self.actor.eval()
                self.target_actor.eval()
                values_one = self.critic_one(states, actions)
                values_two = self.critic_two(states, actions)

                with torch.no_grad():   #   future values
                    actions_ = self.choose_actions(states_, noisy=True, noise_mag=self.TARGET_NOISE_MAG, target=True)

                    values_one_ = self.target_critic_one(states_, actions_)
                    values_two_ = self.target_critic_two(states_, actions_)

                    values_ = torch.min(values_one_, values_two_)
                    values_[dones] = 0.0

                    target = rewards + self.GAMMA * values_ 
                
                critic_one_loss = ((target - values_one)**2).mean()
                critic_two_loss = ((target - values_two)**2).mean()
                critic_losses = critic_one_loss + critic_two_loss
                self.critic_optimizer.zero_grad()
                critic_losses.backward()
                self.critic_optimizer.step()

                self.critic_updates_count += 1

                '''                     actor losses                       '''
                '''        update based on new policy of old states        '''
                if self.critic_updates_count >= self.VALUE_MATURITY_DELAY:
                    if self.critic_updates_count % self.POLICY_DELAY == 0:
                        self.actor.train()
                        self.target_actor.eval()
                        for param in self.critic_params:
                            param.requires_grad = False

                        retrospective_actions = self.choose_actions(states, noisy=False, target=False)
                        retrospective_values = self.critic_one(states, retrospective_actions)
                        actor_loss = -torch.mean(retrospective_values)
                        
                        self.actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor_optimizer.step()

                        for param in self.critic_params:
                            param.requires_grad = True

                        self.update_all_target_params()
                else:
                    print(("Delaying Policy Update: waiting for value maturity "
                        f"(remaining value learn steps: {self.VALUE_MATURITY_DELAY - self.learn_count})"))
        self.learn_count += 1


if __name__ == '__main__':
    env = ContinuousCartPoleEnv()
    agent = Agent(state_shape=(4,), num_actions=1, action_range=1.0)

    high_score = -math.inf
    episode = 0
    num_samples = 0
    while True:
        done = False
        state = env.reset()

        score, frame = 0, 1
        while not done:
            env.render()

            actions = agent.choose_action(state, noisy=True)
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