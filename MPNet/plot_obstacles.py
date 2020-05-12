import numpy as np
import matplotlib.pyplot as plt

N = 30000  # Number of scenarios
NP = 2800  # Number of obstacles
obstacles = np.zeros((N, NP), dtype=np.float32)
for i in range(0, N):
    temp = np.fromfile('/mnt/hgfs/Databases/dataset2/obs_cloud/obc' + str(i) + '.dat')
    temp = temp.reshape(len(temp) // 2, 2)
    plt.plot(temp[:, 0], temp[:, 1], 'ro')
    plt.ylabel('Obstacle map')
    plt.show()
    wait = input("PRESS ENTER TO CONTINUE.")



