from testbench_control import TestBench
import time
import datetime
import cv2
import numpy as np
import sys
from scipy.io import savemat


num_trials = int(sys.argv[1])
radius = int(sys.argv[2])

tb = TestBench('/dev/ttyACM0')
cap = cv2.VideoCapture(0)

while not tb.ready():
    time.sleep(0.1)
    tb.update()

tb.start()

while tb.busy():
    tb.update()

tb.reqData()

x = 1400
y = 6000
z = 0

dx = 0
dy = 0 

mX = 6000
mY = 12000
mZ = 2000

force_threshold = 15

frame_data = []
x_pos = []
y_pos = []
z_pos = []
shape = []
force_thresh = []
force_1 = []
force_2 = []
force_3 = []
force_4 = []

for i in range(num_trials):
    
    p_x = int(np.random.uniform(-radius, radius))
    p_y = int(np.random.uniform(-radius, radius))

    while p_x**2 + p_y**2 > radius**2:
        p_x = int(np.random.uniform(-radius, radius))
        p_y = int(np.random.uniform(-radius, radius))
    
    dx = p_x
    dy = p_y
    
    tb.target_pos(x + dx, y + dy, 0)

    while tb.busy():
        tb.update()
        cap.grab()
    
    tb.press_z(1000, force_threshold)
    while tb.busy():
        tb.update()
        cap.grab()

    time.sleep(0.5)
    data = tb.reqData()
    print(data)

    x_pos.append(x + int(data[data.find('X')+3:data.find('Y')]))
    y_pos.append(y + int(data[data.find('Y')+3:data.find('Z')]))
    shape.append("star")
    force_thresh.append(force_threshold)
    data = data[data.find('Z') + 3:]
    z_pos.append(int(data[:data.find(' ')]))
    ti = data.find(":")
    data = data[ti+2:]
    force_1.append(float(data[:data.find(' ')]))
    data = data[data.find(' ') + 4:]
    force_2.append(float(data[:data.find(' ')]))
    data = data[data.find(' ') + 4:]
    force_3.append(float(data[:data.find(' ')]))
    data = data[data.find(' ') + 4:]
    force_4.append(float(data[:data.find(' ')]))
     
    ret, frame = cap.read()
    cv2.imwrite("cap_frame" + str(i) + ".png", frame)
    frame_data.append(np.copy(frame))

    tb.reset_z()

    while tb.busy():
        tb.update()
        cap.grab()

ctimestr = datetime.datetime.now().strftime("%Y-%m-%d:%H:%M:%S") 

print(x_pos)
print(y_pos)
print(z_pos)
print(shape)
print(force_thresh)
print(force_1)
print(force_2)
print(force_3)
print(force_4)

savemat(ctimestr + 'star_num_data.mat',
                {
                    "x": x_pos,
                    "y": y_pos,
                    "z": z_pos,
                    "shape": shape,
                    "force_thresh": force_thresh,
                    "force_1": force_1,
                    "force_2": force_2,
                    "force_3": force_3,
                    "force_4": force_4
                })
savemat(ctimestr + 'star_frame_data.mat',
                {
                    "frames": frame_data
                })

tb.reset()

while tb.busy():
    tb.update()
    
         
    

