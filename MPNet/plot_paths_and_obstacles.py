import numpy as np
import matplotlib.pyplot as plt
import os
import time

# Create model directory
output_path = 'output'
if not os.path.exists(output_path):
    os.makedirs(output_path)

N = 30000  # Number of scenarios
NP = 2800  # Number of obstacles
filename = os.path.join(output_path, 'obstacle.npy')
if not os.path.exists(filename):
    obstacles = np.zeros((N, NP//2, 2), dtype=np.float32)
    for i in range(0, N):
        temp = np.fromfile('/mnt/hgfs/Databases/dataset2/obs_cloud/obc' + str(i) + '.dat')
        temp = temp.reshape(len(temp) // 2, 2)
        obstacles[i, :, :] = temp
    np.save(filename, obstacles)
else:
    obstacles = np.load(filename)

N2 = 100  # Number of scenarios
NP2 = 4000  # Number of paths
filename2 = os.path.join(output_path, 'paths.npy')
if not os.path.exists(filename2):
    print("Calculating length of the longest trajectory\n")
    max_length = 0
    path_lengths = np.zeros((N2, NP2), dtype=np.int32)
    for i in range(0, N2):
        for j in range(0, NP2):
            fname = '/mnt/hgfs/Databases/dataset2/e' + str(i) + '/path' + str(j) + '.dat'
            if os.path.isfile(fname):
                path = np.fromfile(fname)
                path = path.reshape(len(path) // 2, 2)
                path_lengths[i][j] = len(path)
                if len(path) > max_length:
                    max_length = len(path)

    print("Reading paths\n")
    paths = np.zeros((N2, NP2, max_length, 2), dtype=np.float32)  ## padded paths
    for i in range(0, N2):
        for j in range(0, NP2):
            fname = '/mnt/hgfs/Databases/dataset2/e' + str(i) + '/path' + str(j) + '.dat'
            if os.path.isfile(fname):
                path = np.fromfile(fname)
                path = path.reshape(len(path) // 2, 2)
                for k in range(0, len(path)):
                    paths[i][j][k] = path[k]
    np.save(filename2, paths)
else:
    paths = np.load(filename2)

# Plot scenarios paths and obstacles.
max_length = paths.shape[2]
for i in range(0, N2):
    path_count = 1
    for j in range(0, NP2):
        path_ind = max_length
        for l in range(0, max_length):  # For every trajectory, check its length
            if np.sum(paths[i, j, l:, :]) == 0:
                path_ind = l - 1
                break
        if path_ind != -1:
            plt.plot(obstacles[i, :, 0], obstacles[i, :, 1], 'ro')
            plt.title('Obstacle and path map')
            plt.plot(paths[i, j, 0:path_ind+1, 0], paths[i, j, 0:path_ind+1, 1], 'bo')
            plt.show()
            save_fig = False
            if save_fig:
                filename3 = os.path.join(output_path, 'map_{:06d}_{:06d}.png'.format(i, j))
                plt.savefig(filename3)
            time.sleep(0.5)
            #wait = input("PRESS ENTER TO CONTINUE.\n")




