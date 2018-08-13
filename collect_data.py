from testbench_control import TestBench
import time
import datetime
import cv2
import numpy as np
import sys
from scipy.io import savemat

# XY coordinates experimentally determined for each of the shapes
SHAPE_POS = {'star': {'x': 1384, 'y': 6073}}

#Ex: python collect_data.py star 100 80 3
shape_name = sys.argv[1]
num_trials = int(sys.argv[2])
radius = int(sys.argv[3])
force_inc = int(sys.argv[4])

assert shape_name in SHAPE_POS, "Invalid shape!"

tb = TestBench('/dev/ttyACM0')
cap = cv2.VideoCapture(0)

while not tb.ready():
    time.sleep(0.1)
    tb.update()

tb.start()

while tb.busy():
    tb.update()

print(tb.reqData())

x = SHAPE_POS[shape_name]['x'] 
y = SHAPE_POS[shape_name]['y'] 
z = 0

dx = 0
dy = 0 

mX = 6000
mY = 12000
mZ = 2000

MIN_FORCE_THRESH = 10 
MAX_FORCE_THRESH = 14 

pre_press_frames = []
press_frames = []
x_pos = []
y_pos = []
z_pos = []
shape = []
force_thresh = []
force_1 = []
force_2 = []
force_3 = []
force_4 = []

ctimestr = datetime.datetime.now().strftime("%Y-%m-%d:%H:%M:%S") 

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
    
    time.sleep(0.25)

    ret, frame = cap.read()
    cv2.imwrite("cap_framebefore" + str(i) + ".png", frame)
    ppf = np.copy(frame)

    force_threshold = MIN_FORCE_THRESH 

    # Initial press -- go fast on first 1000 stepss
    
    while force_threshold < MAX_FORCE_THRESH:

        if force_threshold == MIN_FORCE_THRESH: 
            tb.press_z(1200, force_threshold)
        else:
            tb.press_z(0, force_threshold)

        while tb.busy():
            tb.update()
            cap.grab()

        time.sleep(0.5)
        data = tb.reqData()
        print(data)

        x_pos.append(int(data[data.find('X')+3:data.find('Y')]))
        y_pos.append(int(data[data.find('Y')+3:data.find('Z')]))
        shape.append(shape_name)
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
        cv2.imwrite("cap_frame" + str(i) + 'f=' + str(force_threshold) + ".png", frame)
        pre_press_frames.append(np.copy(ppf))
        press_frames.append(np.copy(frame))

        force_threshold += force_inc 
        
    if i % 10 == 0: # Save progress often so we don't lose data!
        savemat('data/' + ctimestr + '-' + shape_name + '.mat',
                {
                    "x": x_pos,
                    "y": y_pos,
                    "z": z_pos,
                    "shape": shape,
                    "force_thresh": force_thresh,
                    "force_1": force_1,
                    "force_2": force_2,
                    "force_3": force_3,
                    "force_4": force_4,
                    "press_frames": press_frames,
                    "pre_press_frames" : pre_press_frames 
                })

    tb.reset_z()

    while tb.busy():
        tb.update()
        cap.grab()


# print(x_pos)
# print(y_pos)
# print(z_pos)
# print(shape)
# print(force_thresh)
# print(force_1)
# print(force_2)
# print(force_3)
# print(force_4)
# 
savemat('data/' + ctimestr + '-' + shape_name + '.mat',
        {
            "x": x_pos,
            "y": y_pos,
            "z": z_pos,
            "shape": shape,
            "force_thresh": force_thresh,
            "force_1": force_1,
            "force_2": force_2,
            "force_3": force_3,
            "force_4": force_4,
            "press_frames": press_frames,
            "pre_press_frames" : pre_press_frames 
        })

tb.reset()

while tb.busy():
    tb.update()
    
         
