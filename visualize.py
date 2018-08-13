import matplotlib.pyplot as plt
from scipy.io import loadmat
import sys

fname = sys.argv[1]

data = loadmat(fname)
press_frames = data['press_frames']
pre_press_frames = data['pre_press_frames']
x = data['x']
y = data['y']
thresh = data['force_thresh']
force1 = data['force_1']
force2 = data['force_2']
force3 = data['force_3']
force4 = data['force_4']

plt.scatter(x, y)
plt.show()

plt.figure(1)
plt.subplot(221)
plt.hist(force1)
plt.subplot(222)
plt.hist(force2)
plt.subplot(223)
plt.hist(force3)
plt.subplot(224)
plt.hist(force4)
plt.show()

print(thresh.shape)

for i in range(0, 3, 3):
    plt.figure(1)
    plt.subplot(311)
    plt.imshow(press_frames[i])
    plt.subplot(312)
    plt.imshow(press_frames[i+1])
    plt.subplot(313)
    plt.imshow(press_frames[i+2])
    plt.show()


