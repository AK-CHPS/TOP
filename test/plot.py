import numpy as np
import random
import math
import sys
import struct
import os.path
import seaborn as sns
import matplotlib.pyplot as plt

res = np.fromfile("res.raw", dtype=np.float64).reshape(202 * 9, 162)
test = np.fromfile("test.raw", dtype=np.float64).reshape(202 * 9, 162)

delta = np.abs(res - test)

sns.heatmap(delta, cmap="YlOrRd")

plt.show()