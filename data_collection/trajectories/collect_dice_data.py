from testbench_control import TestBench
import save_dice_traj
import time
import datetime
import numpy as np
import sys
import cv2
import pickle as pkl
import serial
#from notify_run import Notify

side_camera_index = 2
tb_camera_index = 0

tb = TestBench('/dev/ttyACM0', tb_camera_index, side_camera_index)
resetter = serial.Serial('/dev/ttyUSB0', baudrate=9600, timeout=1)

#notify = Notify()
#notify.register()

while not tb.ready():
    time.sleep(0.1)
    tb.update()

tb.flip_x_reset()
time.sleep(0.5)

tb.start()

while tb.busy():
    tb.update()

ZERO_POS = [5200, 5300, 0]
max_force = 15 
min_force = 6.5 

small_w = 64
small_h = 48

ctimestr = datetime.datetime.now().strftime("%Y-%m-%d:%H:%M:%S")

maxX, maxY, maxZ = 5800, 6100, 1050 
minX, minY, minZ = 4000, 4300, 0

print(tb.req_data())

def reset_dice():
    resetter.write(b'50\n')


def loosen_dice():
    resetter.write(b'2000\n') 

def random_actions(state):
     
    act = [np.random.random_integers(-150, 150), np.random.random_integers(-150, 150), np.random.random_integers(-10, 10)]
    return act

def get_randomoffset():
    return [np.random.random_integers(-10, 10), np.random.random_integers(-10, 10), np.random.random_integers(0, 0)]


def run_traj(num_steps, policy):
    reset_dice()
    time.sleep(1)
    loosen_dice() 
    confirm = ''
    for i in range(resetter.inWaiting()):
        ch = resetter.read().decode()
        confirm += ch
    print(confirm)
    #if confirm == '':
        #notify.send('something happened.. check robot!!')
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

    OFFSET_HOME_POS = pos[:]

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
    
    tb.press_z(600, 7)
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
    action_repeat_count = 0
    action_repeat = 3

    for n in range(num_steps):
        if not action_repeat_count:
            # If action repeat is over, grab next move to take
            #if action_queue:
            #    act = action_queue.pop(0)
                # Actions popped off the queue are not repeated. If repeating
                # is desired, add the action multiple times.
            #else:
            act = policy(pos) 
            action_repeat_count = action_repeat - 1
        else:
            action_repeat_count -= 1
        pos = [pos[i]+act[i] for i in range(3)]  
        if corr_next:
            pos[2] -= 15
        normalize_pos(pos)
        
        tb.target_pos(*pos)
        bt = millis()
        
        while tb.busy():
            tb.update()
        
        print(millis() - bt)
        data = tb.req_data()

        # After every 3 actions (x 3 action repeat), backtrack to relocate the dice

        frame = tb.get_frame()
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
        if (max(forces) < min_force):
            print("Slip detected")
            slip = True
        data['slip'] = slip 

        full_images.append(frame)
        side_images.append(side_frame)
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

    traj = run_traj(18, random_actions)
    
    save_dice_traj.save_tf_record('traj_data/' + ctimestr + '/traj'+str(i) + '/', 'traj' + str(i), traj, mean, std)
    # Save videos
    #save_dice_traj.save_dd_record('traj_data/' + ctimestr + '/traj'+str(i) + '/', 'traj' + str(i), traj)

tb.reset()
while tb.busy():
    tb.update()    

