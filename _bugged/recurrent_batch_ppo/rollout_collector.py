import math
import random

import gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from multiprocessing_env import SubprocVecEnv

class RolloutCollector:
    def __init__(self, num_env_workers, make_env_func, agent, batch_size, rollout_length, num_recurrence_steps,
            state_shape, action_shape, stats):
        ''' -one agent is assigned to a collector. 
            -a collector runs a bunch of envs in paralel to feed to that agent
            -you could run a bunch of collectors simultaniously, 
                |-  and then use weight mixing on the agents seperately
        '''
        self.num_env_workers = num_env_workers
        self.envs = SubprocVecEnv([make_env_func() for i in range(num_env_workers)])
        self.agent = agent
        self.batch_size = batch_size
        self.rollout_length = rollout_length
        self.num_recurrence_steps=num_recurrence_steps
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.stats = stats

        self.buffer_full = False
        self.GAE_calculated = False

        self.gamma  = 0.8
        self.tau    = 0.8

        self.rollout_indices = np.zeros(batch_size)
        self.buffer_width = self.rollout_length + self.num_recurrence_steps
        self.states     = torch.zeros((batch_size, self.buffer_width+1, *state_shape ),   dtype=torch.float32).to(self.agent.device)
        self.actions    = torch.zeros((batch_size, self.buffer_width+1, *action_shape),   dtype=torch.float32).to(self.agent.device)
        self.log_probs  = torch.zeros((batch_size, self.buffer_width+1, *action_shape),   dtype=torch.float32).to(self.agent.device)
        self.values     = torch.zeros((batch_size, self.buffer_width+1,  1           ),   dtype=torch.float32).to(self.agent.device)
        self.rewards    = torch.zeros((batch_size, self.buffer_width+1,  1           ),   dtype=torch.float32).to(self.agent.device)
        self.done_masks = torch.zeros((batch_size, self.buffer_width+1,  1           ),   dtype=torch.float32).to(self.agent.device)
        self.advantages = torch.zeros((batch_size, self.buffer_width+1,  1           ),   dtype=torch.float32).to(self.agent.device)
        self.returns    = torch.zeros((batch_size, self.buffer_width+1,  1           ),   dtype=torch.float32).to(self.agent.device)

        self.state = self.envs.reset()
        self.hidden_state    = torch.zeros((1, self.num_env_workers, self.agent.hidden_state_size)).to(self.agent.device)
        self.cell_state      = torch.zeros((1, self.num_env_workers, self.agent.hidden_state_size)).to(self.agent.device)

    def collect_samples(self):
        if self.buffer_full:
            raise Exception("tried to collect more samples when buffer already full")
        num_runs_to_full = math.ceil(self.batch_size / self.num_env_workers)
        with torch.no_grad():
            
            self.hidden_state    = torch.zeros((1, self.num_env_workers, self.agent.hidden_state_size)).to(self.agent.device)
            self.cell_state      = torch.zeros((1, self.num_env_workers, self.agent.hidden_state_size)).to(self.agent.device)

            for collection_run in range(num_runs_to_full):
                start_index = collection_run * self.num_env_workers
                end_index_exclusive = min(start_index + self.num_env_workers, self.batch_size)
                run_indices = np.arange(start_index, end_index_exclusive)
                worker_indices = run_indices % self.num_env_workers

                for rollout_idx in range(self.buffer_width+1):
                    state = torch.FloatTensor(self.state).to(self.agent.device)
                    
                    #   for recurrences
                    lstm_input = state.view(-1, 1, *self.state_shape)
                    output, (hidden, cell) = self.agent.lstm(lstm_input, (self.hidden_state, self.cell_state))
                    output = output.reshape(self.num_env_workers, self.agent.hidden_state_size)

                    policy_dist = self.agent.actor(output)
                    action = policy_dist.sample()
                    action = action.clamp(-1, 1)    #   depends on env
                    state_, reward, done, info = self.envs.step(action.cpu().numpy())

                    value = self.agent.critic(output)
                    log_prob = policy_dist.log_prob(action)

                    self.states[run_indices, rollout_idx]       = torch.FloatTensor( state[worker_indices]       )
                    self.actions[run_indices, rollout_idx]      = torch.FloatTensor( action[worker_indices]      ) 
                    self.log_probs[run_indices, rollout_idx]    = torch.FloatTensor( log_prob[worker_indices]    ) 
                    self.values[run_indices, rollout_idx]       = torch.FloatTensor( value[worker_indices]       )
                    self.rewards[run_indices, rollout_idx]      = torch.FloatTensor( reward[worker_indices]      ).unsqueeze(1)
                    self.done_masks[run_indices, rollout_idx]   = torch.FloatTensor( 1.0 - done[worker_indices]  ).unsqueeze(1)

                    self.hidden_state[0, worker_indices] *= self.done_masks[run_indices, rollout_idx].expand(-1, self.agent.hidden_state_size)
                    self.cell_state[0, worker_indices]   *= self.done_masks[run_indices, rollout_idx].expand(-1, self.agent.hidden_state_size)
                    self.state = state_

        self.buffer_full = True
        self.stats.update_collection_stats(
            num_samples_collected_inc=self.batch_size * self.rollout_length)

    def compute_gae(self):
        if not self.buffer_full:
            raise Exception("buffer is not full of new samples yet (so not ready for GAE)")

        gae = torch.zeros((self.batch_size, 1)).to(self.agent.device)
        for i in reversed(range(self.buffer_width)):
            delta = self.rewards[:, i] + self.gamma * self.values[:, i+1] * self.done_masks[:, i] - self.values[:, i]
            gae = delta + self.gamma * self.tau * self.done_masks[:, i] * gae
            self.returns[:, self.buffer_width - i] = gae + self.values[:, i]
            self.advantages[:, self.buffer_width - i] = gae

        self.GAE_calculated = True

    def random_batch_iter(self):
        if not self.buffer_full and not self.GAE_calculated:
            raise Exception("buffer is not ready for sampling yet. (not full/no GAE)")

        '''-theres no way all the workers are aligned, especially after an episode or so. 
            so we might just be able to use a vertical index'''
        batch_indices = torch.randperm(self.rollout_length)
        
        #   recurrence stuff
        if self.num_recurrence_steps > 0:
            batch_indices = torch.randperm(self.rollout_length) + self.num_recurrence_steps
            self.hidden_state   = torch.zeros((1, self.batch_size, self.agent.hidden_state_size)).to(self.agent.device)
            self.cell_state     = torch.zeros((1, self.batch_size, self.agent.hidden_state_size)).to(self.agent.device)
        
        for i in range(self.rollout_length):
            index     = batch_indices[i]
            
            #   recurrence stuff
            leading_state_indices = torch.arange(self.num_recurrence_steps) - self.num_recurrence_steps + 1 + index
            leading_states  = self.states[:, leading_state_indices]
            output, (hidden, cell) = self.agent.lstm(leading_states, (self.hidden_state, self.cell_state))
            state = output[:, -1, :]
            action    = self.actions[:, index]
            log_prob  = self.log_probs[:, index]
            advantage = self.advantages[:, index]
            return_   = self.returns[:, index]
            yield state, action, log_prob, advantage, return_

    def reset(self):
        self.buffer_full = False
        self.GAE_calculated = False

        