import numpy
import torch

class DummyEnv:
    def reset(self):
        pass

    def step(self):
        reward = 0
        state_ = numpy.random((64, 5))
        done = False
        info = None

        return reward, state, done, info

    