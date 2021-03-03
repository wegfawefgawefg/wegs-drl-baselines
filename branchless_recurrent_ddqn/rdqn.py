import math
import random
from collections import deque

import gym       
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from agent import Agent
from stats import Stats
from memory_slice import MemorySlice

STATE_SHAPE = (4,)
NUM_ACTIONS = 2
ACTION_SHAPE = (1,)

SLICE_SIZE = 4
# NUM_SAMPLES = 8                                    # collected between each learn step
# NUM_SLICES = math.ceil(NUM_SAMPLES / SLICE_SIZE)    # collected between each learn step

NUM_SLICES = 2
BATCH_SIZE = 3

def collect_slices(stats):
    memory_slices = []

    state = env.reset()
    hidden_state = agent.net.get_batch_hidden_state(batch_size=1).to(agent.device)
    score = 0
    done = False

    slice_backbuffer = deque(maxlen=SLICE_SIZE)
    with torch.no_grad():
        for slice_num in range(NUM_SLICES):
            memory_slice = MemorySlice(SLICE_SIZE, STATE_SHAPE, ACTION_SHAPE)
            for i in range(SLICE_SIZE):
                action, hidden_state_ = agent.choose_action(state, hidden_state)
                state_, reward, done, info = env.step(action)

                memory_slice.store_transition(state, action, reward, state_, done)

                transition = state, action, reward, state_, done
                slice_backbuffer.append(transition)

                state = state_
                hidden_state = hidden_state_
                
                score += reward

                stats.num_samples += 1
                stats.epsilons.append(agent.epsilon.value())

                if done:
                    #   update stats
                    stats.high_score = max(stats.high_score, score)
                    stats.scores.append(score)
                    stats.num_episodes += 1
                    # stats.print_episode_end()

                    #   build retroactive done-ended slice
                    memory_slice = MemorySlice(SLICE_SIZE, STATE_SHAPE, ACTION_SHAPE)
                    for transition in slice_backbuffer:
                        memory_slice.store_transition(*transition)

                    #   reset for next slice
                    state = env.reset()
                    hidden_state = agent.net.get_batch_hidden_state(batch_size=1).to(agent.device)
                    score = 0
                    done = False

                    break

            memory_slices.append(memory_slice)
    return memory_slices

def play_test_episode(stats):
    with torch.no_grad():
        hidden_state = agent.net.get_batch_hidden_state(batch_size=1).to(agent.device)
        state = env.reset()

        score = 0
        frame = 1
        done = False
        while not done:
            env.render()

            action, hidden_state_ = agent.choose_action(state, hidden_state)
            state_, reward, done, info = env.step(action)

            state = state_
            hidden_state = hidden_state_

            score += reward
            frame += 1

        stats.high_score = max(stats.high_score, score)
        stats.scores.append(score)
        stats.num_episodes +=1
        stats.print_episode_end()

if __name__ == '__main__':
    env = gym.make('CartPole-v1').unwrapped
    agent = Agent(learn_rate=0.0001, 
        state_shape=STATE_SHAPE, num_actions=NUM_ACTIONS, action_shape=ACTION_SHAPE,
        batch_size=BATCH_SIZE, slice_size=SLICE_SIZE)
    stats = Stats()

    while True:
        memory_slices = collect_slices(stats)
        for memory_slice in memory_slices:
            agent.slice_replay_buffer.store_memory_slice(*memory_slice.get())
        agent.learn(stats)
        play_test_episode(stats)

