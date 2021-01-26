import math
import random

import gym       
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from memory_slice import MemorySlice

class MemorySliceReplayBuffer:
    def __init__(self, size, slice_size, state_shape, action_shape):
        self.size = size
        self.count = 0

        self.state_memory       = np.zeros((self.size, slice_size,  *state_shape),  dtype=np.float32)
        self.action_memory      = np.zeros((self.size, slice_size, *action_shape),  dtype=np.long)
        self.reward_memory      = np.zeros((self.size, slice_size,             1),  dtype=np.float32)
        self.next_state_memory  = np.zeros((self.size, slice_size,  *state_shape),  dtype=np.float32)
        self.done_memory        = np.zeros((self.size, slice_size,             1),  dtype=np.bool   )

    def store_memory_slice(self, state_slice, action_slice, reward_slice, next_state_slice, done_slice):
        index = self.count % self.size 
        
        self.state_memory[index]      = state_slice
        self.action_memory[index]     = action_slice
        self.reward_memory[index]     = reward_slice
        self.next_state_memory[index] = next_state_slice
        self.done_memory[index]       = done_slice

        self.count += 1

    def sample(self, sample_size, device):
        highest_index = min(self.count, self.size)
        indices = np.random.choice(highest_index, sample_size, replace=False)

        states      = self.state_memory[indices]
        actions     = self.action_memory[indices]
        rewards     = self.reward_memory[indices]
        next_states = self.next_state_memory[indices]
        dones       = self.done_memory[indices]

        states      = torch.tensor( states,        dtype=torch.float32 ).detach().to(device)
        actions     = torch.tensor( actions,       dtype=torch.long ).detach().to(device)
        rewards     = torch.tensor( rewards,       dtype=torch.float32 ).detach().to(device)
        next_states = torch.tensor( next_states,   dtype=torch.float32 ).detach().to(device)
        dones       = torch.tensor( dones,         dtype=torch.bool    ).detach().to(device)

        return states, actions, rewards, next_states, dones

if __name__ == "__main__":
    STATE_SHAPE = (4,)
    ACTION_SHAPE = (1,)
    SLICE_SIZE = 8
    BATCH_SIZE = 8

    replay_buffer = MemorySliceReplayBuffer(
        size=10_000, 
        slice_size=SLICE_SIZE, 
        state_shape=STATE_SHAPE, 
        action_shape=ACTION_SHAPE
    )

    #   create slices
    for i in range(BATCH_SIZE):
        mem_slice = MemorySlice(
            slice_size=SLICE_SIZE, 
            state_shape=STATE_SHAPE, 
            action_shape=ACTION_SHAPE
        )
        for j in range(SLICE_SIZE):
            state = np.ones(STATE_SHAPE)
            action = np.ones(ACTION_SHAPE)
            reward = i * 10 + j
            next_state = np.ones(STATE_SHAPE)
            done = False

            mem_slice.store_memory(state, action, reward, next_state, done)

        #   store slices in memory
        replay_buffer.store_memory(
            *mem_slice.get())

    #   sample memory
    states_rollout_batch, actions_rollout_batch, rewards_rollout_batch, next_states_rollout_batch, dones_rollout_batch = replay_buffer.sample(8)

    print(rewards_rollout_batch.shape)
    print(rewards_rollout_batch)

    for i in range(BATCH_SIZE):
        rewards = rewards_rollout_batch[:, i]
        print(rewards.shape)
        print(rewards)
        break