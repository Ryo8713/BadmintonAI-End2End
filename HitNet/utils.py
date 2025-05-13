import matplotlib.pyplot as plt
import math
import sys
import os

# Add the directory containing the ai_badminton package to the Python path

sys.path.append(os.path.abspath('../python/ai-badminton/src/ai_badminton'))

from ai_badminton.trajectory import Trajectory
from ai_badminton.pose import Pose
from ai_badminton.court import Court, read_court

import numpy as np
import random
from scipy.ndimage.interpolation import shift
from skimage.transform import rescale, resize

def read_coordinate(filename):
    file = open(filename, 'r')
    coordinates = [[float(x) for x in line.split(';')] for line in file]
    return coordinates

def visualize(x, y):
    print(x.shape, y.shape)
    cdict = {0: 'red', 1: 'blue', 2: 'green'}
    plt.figure()
    for g in np.unique(y):
        ix = np.where(y == g)
        plt.scatter(*x[ix, -4:, :].T, c=cdict[g], label=g)
        plt.scatter(*x[ix, 0, :].T, c=cdict[g], label=g)
    plt.show()
    
def resample(series, s):
    flatten = False
    if len(series.shape) == 1:
        series.resize((series.shape[0], 1))
        series = series.astype('float64')
        flatten = True
    series = resize(
        series, (int(s * series.shape[0]), series.shape[1]),
    )
    if flatten:
        series = series.flatten()
    return series   

eps = 1e-6
def reflect(x):
    x = np.array(x)
    idx = np.abs(x) < eps
    for i in range(0, x.shape[1], 2):
        x[:, i] = -x[:, i]
    x[idx] = 0.
    return x

# Identify first hit by distance to pose
# and then alternate hits
def dist_to_pose(pose, p):
    pose = pose.reshape(17, 2) # 17->15 keypoints
    p = p.reshape(1, 2)
    D = np.sum((pose - p) * (pose - p), axis=1)
    return min(D)

def scale_data(x):
    x = np.array(x)
    def scale_by_col(x, cols):
        x_ = np.array(x[:, cols])
        idx = np.abs(x_) < eps
        m, M = np.min(x_[~idx]), np.max(x_[~idx])
        x_[~idx] = (x_[~idx] - m) / (M - m) + 1
        x[:, cols] = x_
        return x

    even_cols = [2*i for i in range(x.shape[1] // 2)]
    odd_cols = [2*i+1 for i in range(x.shape[1] // 2)]
    x = scale_by_col(x, even_cols)
    x = scale_by_col(x, odd_cols)
    return x

identity = lambda x: x
def drop_consecutive(x, rep_value=0.):
    x = np.array(x)
    for i in range(x.shape[0]):
        j = random.randint(0, num_consec-1)
        x[i][max(0, 78*(j-2)):min(78*(j+2), 78*num_consec)] = rep_value
    return x

def corrupt_consecutive(x, rep_value=0.):
    x = np.array(x)
    for i in range(x.shape[0]):
        j = random.randint(0, num_consec-1)
        l, r = max(0, 78*(j-2)), min(78*(j+2), 78*num_consec)
        x[i][l:r] = np.random.rand(1, r-l)
    return x

def drop_data(x, rep_value=0, keep_prob=0.95):
    x = np.array(x)
    # Corrupt 15% of the data
    indices = np.random.choice(
        np.arange(x.size), replace=False,
        size=int(x.size * (1 - keep_prob))
    )
    x[np.unravel_index(indices, x.shape)] = rep_value
    return x

def corrupt_data(x, keep_prob=0.95):
    x = np.array(x)
    idx = np.abs(x) < eps
    # Corrupt 15% of the data
    indices = np.random.choice(
        np.arange(x.size), replace=False,
        size=int(x.size * (1 - keep_prob))
    )
    shape = x[np.unravel_index(indices, x.shape)].shape
    low, hi = max(np.min(x[:,0::2]), np.min(x[:,1::2])), min(np.max(x[:,0::2]), np.max(x[:,1::2]))
    target = np.random.rand(*shape) * (hi-low) + low
    x[np.unravel_index(indices, x.shape)] = target
    x[idx] = 0.
    return x

def jiggle_and_rotate(x):
    # Randomly shift by a vector in [0, 30]
    # and rotate by a random amount between -10 and 10 degrees
    x = np.array(x.reshape((x.shape[0], x.shape[1] // 2, 2)))
    idx = np.abs(x) < eps
    # shift does nothing when we rescale after
    # shift = np.random.rand(1, 2) * 30
    angle = (np.random.rand() - 0.5) * math.pi / 180 * 30
    rotate = np.array([[math.cos(angle), -math.sin(angle)], 
                        [math.sin(angle), math.cos(angle)]])
    a, b = np.random.rand() * 0.1, np.random.rand() * 0.1
    shear = np.array([[1+a*b, a], 
                        [b,     1]])
    x = x @ shear @ rotate
    x[idx] = 0.
    x = x.reshape((x.shape[0], x.shape[1] * 2))
    return x