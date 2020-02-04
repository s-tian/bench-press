from tb_control.testbench_control import TestBench
import time
import datetime
import cv2
import numpy as np
import sys
from scipy.io import savemat
import numpy as np
from tb_control.dynamixel_interface import Dynamixel
import argparse
import ipdb
import yaml
import os

with open('config_extrusion_2cam_large.yaml', 'r') as f:
    config = yaml.load(f)

HOME_POS = config['home']
dyna_config = config['dynamixel']

'''
Command line arg format:
python collect_data.py [num_presses] [offset_radius] [images_per_press] [min_force] [max_force] [--output_dir]
Ex: python collect_data.py star 100 5 7 19

Presses at 100 uniformly randomly selected locations in a radius of 80 steps
centered at the star, taking an image at each of 5 uniformly randomly selected
force thresholds based on load cells from minimum to maximum force threshold.
'''

parser = argparse.ArgumentParser(description='Collect gelsight test data')
parser.add_argument('num_trials', metavar='N', type=int, help='number of presses to collect')
parser.add_argument('--out', metavar='out', type=str, default='data/', help='dir for output data')

args = parser.parse_args()

num_trials = args.num_trials
out = args.out

tb = TestBench('/dev/ttyACM0', 0, 2)
dyna = Dynamixel('/dev/ttyUSB0', dyna_config['home'])

dyna.move_to_angle(-90)
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

x = HOME_POS['x']
y = HOME_POS['y']
z = HOME_POS['z']

dx = 0
dy = 0

mX = 6000
mY = 12000
mZ = 1500 

MIN_FORCE_THRESH = 7 
MAX_FORCE_THRESH = 15

NEW_FILE_EVERY = 30 
data_file_num = 0
dynamixel_angle_min = dyna_config['min-angle']
dynamixel_angle_max = dyna_config['max-angle']

pre_press_frames = []
pre_press_frames_2 = []
press_frames = []
press_frames_2 = []
x_pos = []
y_pos = []
z_pos = []
force_1 = []
force_2 = []
force_3 = []
force_4 = []
contact_angle = []

ctimestr = datetime.datetime.now().strftime("%Y-%m-%d:%H:%M:%S")
data_dir = out + ctimestr
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

def meanwoutliers(data):
    data = [i for i in data if i < 30]
    if len(data) > 2:
        std = np.std(data)  
        m = np.mean(data)
        print(f'std: {std}, mean: {m}')
        data = [i for i in data if (i - m) < 1.2*std]
    print(data)
    return np.mean(data)

with open(data_dir + '/config.yaml', 'w') as outfile:
    yaml.dump(config, outfile)

time.sleep(0.5) 

for i in range(num_trials):
    print('----------------------- Trial {} -----------------------'.format(i)) 
    target_x, target_y, target_z = HOME_POS['x'], HOME_POS['y'], HOME_POS['z']
    x, y, z = target_x, target_y, target_z 
    
    dyna.move_to_angle(-90)
    tb.target_pos(target_x, target_y, target_z)
    while tb.busy():
        tb.update()
    
    time.sleep(1)

    offset_angle = np.random.uniform(-dynamixel_angle_max, -dynamixel_angle_min)
    #angles = np.linspace(60, 90, num=6) 
    #offset_angle = -angles[i]

    total_distance = 18.5
    total_ticks = total_distance / 0.04
    
    y_offset = total_ticks * ((dynamixel_angle_max + offset_angle) / (dynamixel_angle_max - dynamixel_angle_min))
    target_y += y_offset
    tb.target_pos(target_x, target_y, target_z)
    while tb.busy():
        tb.update()
    
    print(offset_angle)
    dyna.move_to_angle(offset_angle)

    # Grab before pressing image
    frame, frame2 = tb.get_frame()
    # cv2.imwrite("cap_framebefore" + str(i) + ".png", frame)

    ppf, ppf2 = np.copy(frame), np.copy(frame2)
    
    while tb.busy():
        tb.update()

    force_min = 0 
    mean = 0
    # arm_length = 82 + (offset_angle/22.5) * 8 
    # max_z = (82 + 2.5 - arm_length * np.cos(offset_angle * np.pi / 180.0)) / 0.04 
    # max_z += (-offset_angle - 22.5) * 15 
    # if offset_angle > -25:
    #     max_z += 50
    # if offset_angle < -55:
    #     max_z += 50
    # max_z += 50 
    #max_z = 60474.06 + (163.777 - 60474.06)/(1+(-offset_angle/402.8657)**1.805)
    slope = (1800-50)/(90-60)
    b = 185 - (slope * 60) 
    max_z = -offset_angle * slope + b
    print('MAX Z' + str(max_z))
    step = 0
    while mean < MIN_FORCE_THRESH: 
        if step == 0:
            target_z += (max_z * 3.0 / 4)
            target_z += (max_z * 1.0/4) % 50
            step += 1 
        else:
            target_z += 50
        if target_z > max_z:
            print('Hit z threshold based on trig!')
            break
        tb.target_pos(target_x, target_y, target_z)
        while tb.busy():
            tb.update()
        data = tb.req_data()
        print(data)
        #force_mean = meanwoutliers([data['force_1'], data['force_2'], data['force_3'], data['force_4']])
        #mean = sum(forces)/len(forces)
        time.sleep(0.2)
        mean = 0
     
    time.sleep(0.5)
    data = tb.req_data()
    print(data)
    
    x_pos.append(data['x'])
    y_pos.append(data['y'])
    z_pos.append(data['z'])
    force_1.append(data['force_1'])
    force_2.append(data['force_2'])
    force_3.append(data['force_3'])
    force_4.append(data['force_4'])
    contact_angle.append(offset_angle)
    
    frame, frame2 = tb.get_frame()
    # cv2.imwrite("cap_frame" + str(i) + 'f=' + str(force_threshold) + ".png", frame)
    pre_press_frames.append(np.copy(ppf))
    pre_press_frames_2.append(np.copy(ppf2))

    press_frames.append(np.copy(frame))
    press_frames_2.append(np.copy(frame2))
    time.sleep(0.5)

    if i % NEW_FILE_EVERY == 0 and i > 0: # Save progress often so we don't lose data!
        savemat(data_dir  + '/data_{}.mat'.format(data_file_num),
                {
                    "x": x_pos,
                    "y": y_pos,
                    "z": z_pos,
                    "force_1": force_1,
                    "force_2": force_2,
                    "force_3": force_3,
                    "force_4": force_4,
                    "contact_angle": contact_angle,
                    "press_frames": press_frames,
                    "press_frames_2": press_frames_2,
                    "pre_press_frames" : pre_press_frames,
                    "pre_press_frames_2" : pre_press_frames_2
                })
        
        data_file_num+=1
        pre_press_frames = []
        pre_press_frames_2 = []
        press_frames = []
        press_frames_2 = []
        x_pos = []
        y_pos = []
        z_pos = []
        force_1 = []
        force_2 = []
        force_3 = []
        force_4 = []
        contact_angle = []

savemat(data_dir + '/data_{}.mat'.format(data_file_num),
        {
            "x": x_pos,
            "y": y_pos,
            "z": z_pos,
            "force_1": force_1,
            "force_2": force_2,
            "force_3": force_3,
            "force_4": force_4,
            "contact_angle": contact_angle,
            "press_frames": press_frames,
            "press_frames_2": press_frames_2,
            "pre_press_frames" : pre_press_frames,
            "pre_press_frames_2" : pre_press_frames_2
        })

dyna.move_to_angle(-90)
