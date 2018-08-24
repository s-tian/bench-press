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
python collect_data.py [shape_name] [num_presses] [offset_radius] [images_per_press] [min_force] [max_force]
Ex: python collect_data.py star 100 5 7 19

Presses at 100 uniformly randomly selected locations in a radius of 80 steps
centered at the star, taking an image at each of 5 uniformly randomly selected
force thresholds based on load cells from minimum to maximum force threshold.
'''

parser = argparse.ArgumentParser(description='Collect gelsight test data')
parser.add_argument('shape_name', metavar='shape', type=str, choices=SHAPE_POS.keys(), help='name of shape')
parser.add_argument('num_trials', metavar='N', type=int, help='number of presses to collect')
parser.add_argument('radius', metavar='radius', type=int, help='radius of circle to uniformly select press location in')
parser.add_argument('num_images', metavar='num_images', type=int, help='number of images to get per press')
parser.add_argument('min_force', metavar='min_force', type=float, help='minimum of the range to select random threshold forces from')
parser.add_argument('max_force', metavar='max_force', type=float, help='maximum of the range to select random threshold forces from')

args = parser.parse_args()

shape_name = args.shape_name
num_trials = args.num_trials
radius = args.radius
images_per_press = args.num_images
min_force_thresh = args.min_force
max_force_thresh = args.max_force

assert shape_name in SHAPE_POS, "Invalid shape!"

tb = TestBench('/dev/ttyACM0', 0)

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

print(tb.req_data())

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

    time.sleep(0.25)

    # Grab before pressing image
    frame = tb.get_frame()
    # cv2.imwrite("cap_framebefore" + str(i) + ".png", frame)
    ppf = np.copy(frame)

    force_thresholds = np.sort(np.random.uniform(min_force_thresh, max_force_thresh,
                                                 size=images_per_press))

    for i, force_threshold in enumerate(force_thresholds):

        # Initial press -- go fast on first 1200 stepss
        if i == 0:
            tb.press_z(1200, force_threshold)
        else:
            tb.press_z(0, force_threshold)

        while tb.busy():
            tb.update()

        time.sleep(0.5)
        data = tb.req_data()
        print(data)

        shape.append(shape_name)
        force_thresh.append(force_threshold)

        x_pos.append(data['x'])
        y_pos.append(data['y'])
        z_pos.append(data['z'])
        force_1.append(data['force_1'])
        force_2.append(data['force_2'])
        force_3.append(data['force_3'])
        force_4.append(data['force_4'])


        frame = tb.get_frame()
        # cv2.imwrite("cap_frame" + str(i) + 'f=' + str(force_threshold) + ".png", frame)
        pre_press_frames.append(np.copy(ppf))
        press_frames.append(np.copy(frame))

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
