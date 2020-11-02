from collections import deque
from matplotlib import pyplot as plt
from skimage import transform
from torch.utils.tensorboard import SummaryWriter
import datetime
import gym       
import math
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agent import Agent

def main():
    TENSOR_BOARD = False
    SAVE_CHECKPOINTS = False
    CHECKPOINT_INTERVAL = 1000
    LR = 0.001

    if TENSOR_BOARD:
        ENV_NAME = "Cartpole"
        DISC_OR_CONT = "Disc"
        ALGO_NAME = "DQN_per_runtime_test"
        # ALGO_NAME = "DQN"

        YMDHMS = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
        CHECKPOINT_NAME = "_".join([ENV_NAME, DISC_OR_CONT, ALGO_NAME, str(LR)])
        RUN_NAME = "_".join([YMDHMS, CHECKPOINT_NAME])
        RUNS_PATH = os.path.join("..", "runs", "pertest3", RUN_NAME)
        writer = SummaryWriter(RUNS_PATH, comment=CHECKPOINT_NAME)

    NUM_EPS = 2000
    NUM_SAMPLES = 10000

    env = gym.make('CartPole-v1').unwrapped
    agent = Agent(lr=LR, state_shape=(4,), num_actions=2, batch_size=64)

    num_hidden_episodes = -1
    high_score = -math.inf

    episode = 0
    num_samples = 0
    while True:
        if episode == NUM_EPS:
            break
        if num_samples > NUM_SAMPLES:
            break

        done = False
        state = env.reset()

        score, frame = 0, 1
        while not done:
            if frame == 5000:
                break
            # if episode > num_hidden_episodes:
            #     env.render()

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
                "score {:12.3f}, epsilon {:6.3f}, beta {:6.3f}").format(
            num_samples, episode, high_score, score, agent.epsilon.value(), 
            agent.importance_exp.value()))

        if SAVE_CHECKPOINTS:
            if episode % CHECKPOINT_INTERVAL == 0:
                print("SAVING CHECKPOMT")
                checkpointName = "%s_%d_%d.dat" % (CHECKPOINT_NAME, episode, score)
                fname = os.path.join('.', 'checkpoints', checkpointName)
                torch.save( agent.dqn.state_dict(), fname )

        episode += 1

        if TENSOR_BOARD:
            writer.add_scalar("ep_score", score, episode)
            writer.flush()

if __name__ == '__main__':
    import timeit
    print(timeit.timeit("main()", setup="from __main__ import main", number=1))