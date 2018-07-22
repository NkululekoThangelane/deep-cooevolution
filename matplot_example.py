import matplotlib.pyplot as plt
import numpy as np
import time

fig, ax = plt.subplots()

tstart = time.time()
num_plots = 0
while time.time()-tstart < 2:
    ax.clear()
    ax.plot(np.random.randn(1000))
    plt.pause(0.001)
    num_plots += 1

plt.savefig("ran.jpg")
print(num_plots)