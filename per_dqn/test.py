# import random
# import numpy as np

# a = np.arange(8)/10.0 + 0.1
# p_total = a.sum()

# print(a)

# batch_size = 4

# every_range_len = p_total / batch_size
# # print(every_range_len)
# for i in range(batch_size):
#     # mass = random.random() * every_range_len + i * every_range_len
#     mass = every_range_len + i * every_range_len
#     print(mass)

# class Lerper:
#     def __init__(self, start, end, num_steps):
#         self.delta = (end - start) / float(num_steps)
#         self.num = start - self.delta
#         self.count = 0
#         self.num_steps = num_steps

#     def value(self):
#         if self.count <= self.num_steps:
#             self.num += self.delta
#         self.count += 1
#         return self.num

# lerper = Lerper(5, -10, 8)
# for i in range(50):
#     print(lerper.value())


import numpy as np
from cpprb import PrioritizedReplayBuffer

buffer_size = 256
state_shape = (4,)

pri_replay_buffer = PrioritizedReplayBuffer(
    buffer_size, 
    {   "obs":      { "shape": state_shape  },
        "act":      { "shape": 1            },
        "rew":      {                       },
        "next_obs": { "shape": state_shape  },
        "done":     {                       }
    },
    alpha = 0.5)


batch_size = 10

for i in range(batch_size):
    pri_replay_buffer.add(
        obs=np.array([i, 0, 0, 0]),
        act=i,
        rew=i,
        next_obs=np.array([i, 0, 0, 0]),
        done=0)   

batch = pri_replay_buffer.sample(batch_size, beta=0.5)

rewards = batch["obs"]

print(batch.keys())

# indices = batch["indexes"]
# weights = batch["weights"]