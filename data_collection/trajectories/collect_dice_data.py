from testbench_control import TestBench
import save_dice_traj
import time
import datetime
import numpy as np
import sys
import matplotlib.pyplot as plt
import cv2
import pickle as pkl
import serial

side_camera_index = 0
tb_camera_index = 1

tb = TestBench('/dev/ttyACM0', tb_camera_index, side_camera_index)
resetter = serial.Serial('/dev/ttyUSB0', baudrate=9600, timeout=1)

while not tb.ready():
    time.sleep(0.1)
    tb.update()

tb.flip_x_reset()
time.sleep(0.5)

tb.start()

while tb.busy():
    tb.update()

ZERO_POS = [4900, 5200, 0]
max_force = 15 
min_force = 6.5 

small_w = 64
small_h = 48

ctimestr = datetime.datetime.now().strftime("%Y-%m-%d:%H:%M:%S")

maxX, maxY, maxZ = 5800, 6100, 1050 
minX, minY, minZ = 4000, 4300, 0

print(tb.req_data())

def reset_dice():
    resetter.write(b'350\n')

def loosen_dice():
    resetter.write(b'4000\n') 

def random_actions(state):
     
    act = [np.random.random_integers(50, 200), np.random.random_integers(50, 200), np.random.random_integers(0, 10)]
    for i in range(3):
        if np.random.rand() > 0.5:
            act[i] *= -1
    return act

def get_randomoffset():
    return [np.random.random_integers(-50, 50), np.random.random_integers(-50, 50), np.random.random_integers(0, 0)]


def run_traj(num_steps, policy):
    reset_dice()
    time.sleep(1)
    loosen_dice() 
    num_corr = 0
    images = []
    full_images = []
    side_images = []
    states = []
    pos = ZERO_POS[:]
    offset = get_randomoffset()

    pos[0] += offset[0]
    pos[1] += offset[1]
    pos[2] += offset[2]

    tb.target_pos(*pos)
    while tb.busy(): tb.update()
    frame, data = tb.get_frame(), tb.req_data()
    time.sleep(0.05)
    
    full_images.append(frame)
         
    side_frame = tb.get_side_cam_frame()
    side_images.append(side_frame)
    
    images.append(cv2.resize(frame, (small_w, small_h)))
    data['x_act'] = 0
    data['y_act'] = 0
    data['z_act'] = 0

    states.append(data)
    
    tb.press_z(600, 5)
    while tb.busy():
        tb.update()
    pos[2] = tb.req_data()['z']
    print('z pos' + str(pos[2]))
    
    while tb.busy():
        tb.update()
    
    def normalize_pos(pos):
        pos[0] = min(maxX, max(minX, pos[0]))
        pos[1] = min(maxY, max(minY, pos[1]))
        pos[2] = min(maxZ, max(minZ, pos[2]))
    
    def millis():
        return int(round(time.time()*1000))

    act = None
    slip = False
    corr_next = False
    action_queue = []

    for n in range(num_steps):

        if n % 5 == 0:
            if action_queue:
                act = action_queue.pop(0)
            else:
                act = policy(pos) 
        pos = [pos[i]+act[i] for i in range(3)]  
        if corr_next:
            pos[2] -= 15
        normalize_pos(pos)
        
        tb.target_pos(*pos)
        bt = millis()
        
        while tb.busy():
            tb.update()
        
        print(millis() - bt)
        frame, data = tb.get_frame(), tb.req_data() 
        side_frame = tb.get_side_cam_frame()
        data['x_act'] = act[0]
        data['y_act'] = act[1]
        data['z_act'] = act[2]

        print(data)
        forces = [data['force_1'], data['force_2'], data['force_3'], data['force_4']]
        avg = sum(forces)/4

        if avg > max_force:
            print('force limit crossed')
            corr_next = True
            num_corr += 1
        else:
            corr_next = False

        # if (max(forces) < min_force):
        #     print("Slip detected") 
        #     slip = True
        #     if not action_queue:
        #         new_act = act[:]
        #         new_act[0] = -new_act[0]
        #         new_act[1] = -new_act[1]
        #         action_queue.append(new_act)
        #         action_queue.append([0, 0, 20 + new_act[2]])
        #         new_act[2] = -15 - new_act[2]
               
        data['slip'] = slip 

        full_images.append(frame)
        side_frames.append(side_frame)
        images.append(cv2.resize(frame, (small_w, small_h)))
 
        states.append(data)
        n += 1

    tb.reset_z()
    while tb.busy():
        tb.update()

    #for i in range(0, len(images), 5):
     #   plt.imshow(images[i])     
     #   plt.show()
    

    #final_image = images[-1]

    print("Corrections: " + str(num_corr))
    return {'images': np.array(images), 'states': np.array(states), 'full_images': np.array(full_images), 'side_images': side_images }    

ctimestr = datetime.datetime.now().strftime("%Y-%m-%d:%H:%M:%S")

with open('dice_stats.pkl', 'rb') as f:
    stats = pkl.load(f)
    mean, std = stats['mean'], stats['std']

for i in range(5000):
    if not i % 100:
        reset_dice()
        tb.reset()
        while tb.busy():
            tb.update()

    traj = run_traj(25, random_actions)
    
    save_dice_traj.save_tf_record('traj_data/' + ctimestr + '/traj'+str(i) + '/', 'traj' + str(i), traj, mean, std)

tb.reset()
while tb.busy():
    tb.update()    

