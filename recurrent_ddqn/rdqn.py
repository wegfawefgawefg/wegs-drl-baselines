import math
import random

import gym       
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from agent import Agent
from trajectory import Trajectory

if __name__ == '__main__':
    env = gym.make('CartPole-v1').unwrapped
    agent = Agent(learn_rate=0.001, input_shape=(4,), num_actions=2)

    high_score = -math.inf
    episode = 0
    num_samples = 0
    NUM_EPISODES_PER_TRAIN = 8

    while True:
        '''     GATHER SAMPLES      '''
        with torch.no_grad():
            for i in range(NUM_EPISODES_PER_TRAIN):
                hidden_state = agent.net.get_new_hidden_state().to(agent.device)
                trajectory = Trajectory()

                state = env.reset()

                score = 0
                frame = 1
                done = False
                while not done:
                    action, hidden_state_ = agent.choose_action(state, hidden_state)
                    state_, reward, done, info = env.step(action)
                    trajectory.store_transition(state, action, reward, done)

                    state = state_
                    hidden_state = hidden_state_

                    num_samples += 1
                    score += reward
                    frame += 1

                agent.store_trajectory(trajectory)

                high_score = max(high_score, score)

                print(("total samples: {}, ep {}: high-score {:12.3f}, score {:12.3f}, epsilon {:12.3f}").format(
                    num_samples, episode, high_score, score, agent.epsilon.value()))

                episode += 1

        '''     LEARN   ''' 
        print("train")
        agent.learn()

        '''     TEST    '''
        with torch.no_grad():
            hidden_state = agent.net.get_new_hidden_state().to(agent.device)
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

                num_samples += 1
                score += reward
                frame += 1

            high_score = max(high_score, score)

            print(("total samples: {}, ep {}: high-score {:12.3f}, score {:12.3f}, epsilon {:12.3f}").format(
                num_samples, episode, high_score, score, agent.epsilon.value()))

            episode += 1
