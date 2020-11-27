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
from multiprocessing_env import SubprocVecEnv

'''
TODO:
    -remove weight initing
    -change done masking to be not 1-done
        actual zero masking bc more common idiom
    -make parralelized GAE
    -real numpy replay buffer
    -remove the weird ittering
    -seperate the networks
        then clean up getting state value, vs policy, vs action
    -move around where the training code is
    -add test scoring
'''

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),     nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),     nn.ReLU(),
            nn.Linear(hidden_size, 1))

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),     nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),     nn.ReLU(),
            nn.Linear(hidden_size, num_outputs))

        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        
        self.apply(init_weights)
        
    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu)
        policy_dist  = torch.distributions.Normal(mu, std)
        return policy_dist, value

class Agent():
    def __init__(self, num_inputs, num_actions):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

        self.num_inputs = num_inputs
        self.num_outputs = num_actions
        self.hidden_size = 256
        self.learn_rate = 1e-4
        self.gamma = 0.99
        self.tau = 0.95

        self.epochs_per_train = 4

        self.model = ActorCritic(self.num_inputs, self.num_outputs, self.hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learn_rate)

    def compute_gae(self, next_value, rewards, masks, values):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + self.gamma * self.tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def batch_iter(self, mini_batch_size, states, actions, log_probs, returns, advantage):
        num_states = states.size(0)
        num_mini_batches = num_states // mini_batch_size
        for _ in range(num_mini_batches):
            rand_ids = np.random.randint(0, num_states, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]

    def learn(self, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
        for _ in range(self.epochs_per_train):
            for state, action, old_log_probs, return_, advantage in self.batch_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
                policy_dist, value = self.model(state)
                new_log_probs = policy_dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

                actor_loss  = - torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                entropy = policy_dist.entropy().mean()
                loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()



def collect_new_trajectories(agent, envs, num_steps):
    global frame

    states    = []
    actions   = []
    rewards   = []
    masks     = []
    values    = []
    log_probs = []

    state = envs.reset()
    for _ in range(num_steps):
        state = torch.FloatTensor(state).to(agent.device)

        policy_dist, value = agent.model(state)    #   could be cleaner if the models were seperate
        action = policy_dist.sample()

        next_state, reward, done, _ = envs.step(action.cpu().numpy())
        
        log_prob = policy_dist.log_prob(action)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(agent.device)
        mask = torch.FloatTensor(1 - done).unsqueeze(1).to(agent.device)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        masks.append(mask)
        values.append(value)
        log_probs.append(log_prob)
        
        state = next_state
        frame += 1

    return states, actions, rewards, next_state, masks, values, log_probs

def test_run(agent, env, high_score):
    state = env.reset()

    done = False
    score = 0
    while not done:
        env.render()
        state = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        policy_dist, _ = agent.model(state)
        action = policy_dist.sample().cpu().numpy()[0]
        next_state, reward, done, _ = env.step(action)
        state = next_state
        score += reward

    print(("high-score {:12.3f}, score {:12.3f}").format(
        high_score, score, frame))
    
    high_score = max(score, high_score)
    return high_score

# ENV_NAME = "Pendulum-v0"
ENV_NAME = "LunarLanderContinuous-v2"

def make_env(ENV_NAME):

    def _thunk():
        env = gym.make(ENV_NAME)
        # env = ContinuousCartPoleEnv()
        return env

    return _thunk

if __name__ == "__main__":
    NUM_STEPS_PER_TRAJECTORY = 20
    MINI_BATCH_SIZE = 5
    NUM_ENVS = 10
    
    envs = [make_env(ENV_NAME) for i in range(NUM_ENVS)]
    envs = SubprocVecEnv(envs)
    # NUM_ACTIONS = envs.action_space.shape[0]
    STATE_SIZE = 8
    NUM_ACTIONS = 2

    agent = Agent(num_inputs=STATE_SIZE, num_actions=NUM_ACTIONS)

    test_env = gym.make(ENV_NAME)

    high_score = -math.inf
    frame  = 0
    while True:
        high_score = test_run(agent, test_env, high_score)

        states, actions, rewards, next_state, masks, values, log_probs = collect_new_trajectories(agent, envs, NUM_STEPS_PER_TRAJECTORY)

        next_state = torch.FloatTensor(next_state).to(agent.device)
        _, next_value = agent.model(next_state)
        returns = agent.compute_gae(next_value, rewards, masks, values)

        states    = torch.cat(states)
        actions   = torch.cat(actions)
        log_probs = torch.cat(log_probs).detach()
        
        values    = torch.cat(values).detach()
        returns   = torch.cat(returns).detach()
        advantage = returns - values
        
        agent.learn(MINI_BATCH_SIZE, states, actions, log_probs, returns, advantage)
