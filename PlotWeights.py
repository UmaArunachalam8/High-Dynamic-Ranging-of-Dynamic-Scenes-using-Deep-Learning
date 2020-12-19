import numpy as np 
from matplotlib import pyplot as plt
z = np.arange(0.05, 0.95, 0.01)
u = np.ones((z.size, 1))
t = np.minimum(z, 1 - z)
g = np.exp(-4 * (z - 0.5) ** 2 / 0.25)
p = np.ones((z.size, 1)) * 2

fig = plt.figure()
fig.add_subplot(2, 2, 1)
plt.plot(z, u)
plt.title("Uniform")
fig.add_subplot(2, 2, 2)
plt.plot(z, t)
plt.title("Tent")
fig.add_subplot(2, 2, 3)
plt.plot(z, g)
plt.title("Gaussian")
fig.add_subplot(2, 2, 4)
plt.plot(z, p)
plt.title("Photon")
plt.yticks(color='w')
plt.show()