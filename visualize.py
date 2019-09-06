import matplotlib.pyplot as plt
from scipy.io import loadmat
import sys

fname = sys.argv[1]
data = loadmat(fname)
press_frames = data['press_frames']
pre_press_frames = data['pre_press_frames']
x = data['x']
y = data['y']
z = data['z']

plt.scatter(z, z)
plt.show()

plt.imshow(pre_press_frames[0])
print(pre_press_frames.shape)

for i in range(0, 200, 2):
    plt.show()
    plt.figure(1)
    plt.subplot(411)
    plt.imshow(pre_press_frames[i])
    plt.subplot(412)
    plt.imshow(press_frames[i])
    plt.subplot(413)
    plt.imshow(pre_press_frames[i+1])
    plt.subplot(414)
    plt.imshow(press_frames[i+1])
    plt.show()


