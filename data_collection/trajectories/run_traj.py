from testbench_control import TestBench
from save_traj import save_tf_record
import time
import datetime
import numpy as np
import sys
import matplotlib.pyplot as plt

tb = TestBench('/dev/ttyACM0', 0)

while not tb.ready():
    time.sleep(0.1)
    tb.update()

tb.start()

while tb.busy():
    tb.update()

ZERO_POS = [2650, 6040, 1240]
max_force = 20

print(tb.req_data())

def random_actions(state):
     
    return [np.random.random_integers(-10, 20), np.random.random_integers(-20, 20), np.random.random_integers(-5, 5)]

def run_traj(num_steps, policy):
    images = []
    states = []
    pos = ZERO_POS[:] 
    tb.target_pos(*pos)
    while tb.busy():
        tb.update()
    frame, data = tb.get_frame(), tb.req_data()
    time.sleep(0.05)
    
    images.append(frame) 
    states.append(data)
    
    def normalize_pos(pos):
        mX, mY, mZ = 6000, 12000, 2000
        pos[0] = min(mX, max(0, pos[0]))
        pos[1] = min(mY, max(0, pos[1]))
        pos[2] = min(mY, max(0, pos[2]))

    for n in range(num_steps):
        act = policy(pos) 
        pos = [pos[i]+act[i] for i in range(3)]  
        normalize_pos(pos)
        
        tb.target_pos(*pos)
        
        time.sleep(0.15)
        j = 0
        while tb.busy():
            j += 1
            print('loop ' + str(j))
            tb.update()
        
        frame, data = tb.get_frame(), tb.req_data() 
        avg = (data['force_1'] + data['force_2'] + data['force_3'] + data['force_4']) / 4
        if avg > max_force:
            pos[2] -= 15

        images.append(frame) 
        states.append(data)
    tb.reset()
    while tb.busy():
        tb.update()

    for img in images:
        plt.imshow(img)     
        plt.show()

    return {'images': np.array(images), 'states': np.array(states)}    

for i in range(1):

    traj = run_traj(50, random_actions)
    save_tf_record('traj_data/', 'traj' + str(i), traj)
    

