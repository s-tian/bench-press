from testbench_control import TestBench
import time
import datetime
import cv2
import numpy as np
import sys
from scipy.io import savemat
import argparse

# XY coordinates experimentally determined for each of the shapes
SHAPE_POS = {'star': {'x': 1384, 'y': 6073}, 
             'triangle': {'x': 1400, 'y': 7800},
             'square': { 'x': 2650, 'y': 7780},
             'hemisphere': {'x': 2650, 'y': 6040},
             'cylinder': {'x': 3900, 'y': 7780},
             'side-cylinder': {'x': 3900, 'y': 6020}
            }

'''
Command line arg format:
python collect_data.py [shape_name] [num_presses] [offset_radius] [force_inc]
Ex: python collect_data.py star 100 80 3

Presses at 100 uniformly randomly selected locations in a radius of 80 steps
centered at the star, taking an image at every increment of avg 3 reading 
from load cells from minimum to maximum force threshold, as defined below.
'''

parser = argparse.ArgumentParser(description='Collect gelsight test data')
parser.add_argument('shape_name', metavar='shape', type=str, choices=SHAPE_POS.keys(), help='name of shape')
parser.add_argument('num_trials', metavar='N', type=int, help='number of presses to collect')
parser.add_argument('radius', metavar='radius', type=int, help='radius of circle to uniformly select press location in')
parser.add_argument('force_inc', metavar='increment', type=int, default=3, help='force threshold increment at which to take data while pressing. Determines how many images are taken and at what interval.')

args = parser.parse_args()

shape_name = args.shape_name
num_trials = args.num_trials
radius = args.radius
force_inc = args.force_inc

assert shape_name in SHAPE_POS, "Invalid shape!"

tb = TestBench('/dev/ttyACM0')
cap = cv2.VideoCapture(0)

while not tb.ready():
    time.sleep(0.1)
    tb.update()

tb.start()

while tb.busy():
    tb.update()

'''
Grab a quick reading, use to verify that load cells have been initialized 
and tared correctly
'''

print(tb.reqData())

x = SHAPE_POS[shape_name]['x'] 
y = SHAPE_POS[shape_name]['y'] 
z = 0

dx = 0
dy = 0 

mX = 6000
mY = 12000
mZ = 2000

MIN_FORCE_THRESH = 7 
MAX_FORCE_THRESH = 19 

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
    
    target_x = x + dx
    target_y = y + dy
    
    assert target_x < mX, "Invalid X target: " + str(target_x)
    assert target_y < mY, "Invalid Y target: " + str(target_y)
    
    tb.target_pos(target_x, target_y, 0)

    while tb.busy():
        tb.update()
        cap.grab()
    
    time.sleep(0.25)
    
    # Grab before pressing image
    ret, frame = cap.read()
    # cv2.imwrite("cap_framebefore" + str(i) + ".png", frame)
    ppf = np.copy(frame)

    force_threshold = MIN_FORCE_THRESH 
    
    while force_threshold < MAX_FORCE_THRESH:

        # Initial press -- go fast on first 1200 stepss
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
        # cv2.imwrite("cap_frame" + str(i) + 'f=' + str(force_threshold) + ".png", frame)
        pre_press_frames.append(np.copy(ppf))
        press_frames.append(np.copy(frame))

        force_threshold += force_inc 
        
    if i % 5 == 0: # Save progress often so we don't lose data!
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
