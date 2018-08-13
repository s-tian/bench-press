import matplotlib.pyplot as plt
from scipy.io import loadmat
import sys

fname = sys.argv[1]

data = loadmat(fname)
press_frames = data['press_frames']
pre_press_frames = data['pre_press_frames']

for i in range(5):
    plt.imshow(press_frames[i])


