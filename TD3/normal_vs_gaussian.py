'''The type of noise in TD3 sometimes seems to give drastically different results.
This lets you compare histogram equalized noise with gaussian noise.
Current TD3 implementation is using gaussian noise, torch.randn_like
'''
import numpy as np

mu = 0.0
sigma = 1.0

FLAT_VS_GAUSSIAN = True
if FLAT_VS_GAUSSIAN:    #   flat noise
    s = np.random.normal(mu, sigma, 1000)
else:   #   gaussian noise
    s = np.random.random(1000)

''' just plots the selected noise   '''
print(abs(mu - np.mean(s)))
abs(sigma-np.std(s, ddof=1))

import matplotlib.pyplot as plt
count, bins, ignored = plt.hist(s, 30, density=True)
plt.plot(bins, 
    1/(sigma * np.sqrt(2*np.pi)) * np.exp( -(bins-mu)**2 / (2*sigma**2)),
    linewidth=2, color='r')
plt.show()