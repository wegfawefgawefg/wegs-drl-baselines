import matplotlib.pyplot as plt
from tqdm import tqdm

gamma = 0.99999
total = 0
reward = 1
for i in range(200):
    total += reward
    reward *= gamma
print(total)
