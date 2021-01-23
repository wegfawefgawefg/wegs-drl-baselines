import math
import random

import gym       
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from agent import Agent
from memory import Memory

if __name__ == '__main__':
    env = gym.make('CartPole-v1').unwrapped
    agent = Agent(learn_rate=0.0001, input_shape=(4,), num_actions=2)

    high_score = -math.inf
    episode = 0
    num_samples = 0
    NUM_EPISODES_PER_TRAIN = 32

    while True:
        agent.memories = []
        '''     GATHER SAMPLES      '''
        for i in range(NUM_EPISODES_PER_TRAIN):
            agent.actor.train()
            agent.critic.train()

            actor_hidden_state = agent.actor.get_new_hidden_state()
            critic_hidden_state = agent.critic.get_new_hidden_state()

            memory = Memory()

            state = env.reset()

            score = 0
            frame = 1
            done = False
            while not done:
                env.render()

                action, policy, actor_hidden_state_, action_log_prob = \
                    agent.choose_action(state, actor_hidden_state)
                state_, reward, done, info = env.step(action)
                memory.store(state, action, policy, reward, done, 
                    actor_hidden_state, action_log_prob)

                if i == 0:
                    print(policy)


                state = state_
                actor_hidden_state = actor_hidden_state_

                num_samples += 1
                score += reward
                frame += 1

            agent.store_memory(memory)

            high_score = max(high_score, score)

            print(("total samples: {}, ep {}: high-score {:12.3f}, score {:12.3f}").format(
                num_samples, episode, high_score, score))

            episode += 1

        '''     LEARN   ''' 
        print("train")
        agent.learn()