import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def bn(a, maxVal, minVal):
    for i in range(len(a)):
        a[i] = (a[i] - minVal) / (maxVal - minVal)
    return a


a = np.zeros((2, 5, 5))
a[0, 0, 0] = 10
a[0, 0, 1] = 11
a[1, 1, 1] = 25
a[1, 1, 2] = 26

a[0, 0, 2] = 8
a[0, 0, 3] = 7
a[1, 1, 4] = 5
a[1, 2, 4] = 2


a[a>10] = 10

bn(a, np.max(a), np.min(a))

print(a)



