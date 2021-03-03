import os
import pickle

import numpy
import matplotlib.pyplot as plt
from pprint import pprint

with open('data.pkl', 'rb') as data_file:
    data = pickle.load(data_file)

step = data['step']
scores = data['score']
qs = data['q']

fig, ax = plt.subplots(2)
ax[0].plot(step, scores)
ax[0].set_xlabel('steps')
ax[0].set_ylabel('scores')

ax[1].plot(step, qs)
ax[1].set_xlabel('steps')
ax[1].set_ylabel('q')
plt.show()