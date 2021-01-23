import matplotlib.pyplot as plt
from tqdm import tqdm

total = 0
gammas = [
    0.9, 
    0.99,
    0.999,
    0.9999, 
    0.99999,
    0.999999, 
    0.9999999,
    0.99999999, 
    0.999999999,
]

x = list(range(len(gammas)))
y = []

for gamma in tqdm(gammas):
    total = 0
    reward = 1
    for i in range(10000000):
        total += reward
        reward *= gamma
    print(total)

    y.append(total)

plt.plot(x, y)
plt.show()