import math
import random

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from continuous_cartpole import ContinuousCartPoleEnv
from rollout_collector import RolloutCollector
from agent import Agent
from stats import Stats

def play_test_episode(agent, env, stats):
    with torch.no_grad():
        state = env.reset()
        score = 0
        done = False
        while not done:
            env.render()
            action = agent.act(state, deterministic=True)
            state_, reward, done, info = env.step(action)
            state = state_
            score += reward

    stats.update_test_stats(num_test_episodes_inc=1, latest_test_score=score)
    stats.print_test_run_stats()

def make_env():
    def _thunk():
        env = ContinuousCartPoleEnv()
        return env
    return _thunk

if __name__ == "__main__":
    STATE_SHAPE = (4,)
    ACTION_SHAPE = (1,)
    
    stats = Stats()
    agent = Agent(STATE_SHAPE, ACTION_SHAPE, stats)
    rollout_collector = RolloutCollector(
        num_env_workers=8, make_env_func=make_env, agent=agent, batch_size=64, rollout_length=32,
            state_shape=STATE_SHAPE, action_shape=ACTION_SHAPE, stats=stats)

    test_env = ContinuousCartPoleEnv()
    while True:
        rollout_collector.collect_samples()
        rollout_collector.compute_gae()
        agent.learn(rollout_collector)
        rollout_collector.reset()

        play_test_episode(agent, test_env, stats)