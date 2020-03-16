import datetime
import time

import cv2
import numpy as np
import save_traj
from testbench_control import TestBench

tb = TestBench('/dev/ttyACM0', 0)
while not tb.ready():
    time.sleep(0.1)
    tb.update()

tb.start()

while tb.busy():
    tb.update()

ZERO_POS = [2650, 6040, 870]
max_force = 15
min_force = 5

small_w = 128
small_h = 96

ctimestr = datetime.datetime.now().strftime("%Y-%m-%d:%H:%M:%S")

print(tb.req_data())


def random_actions(state):
    return [np.random.random_integers(-70, 70), np.random.random_integers(-70, 70), np.random.random_integers(-10, 10)]


def run_traj(num_steps, policy):
    num_corr = 0
    images = []
    full_images = []
    states = []
    pos = ZERO_POS[:]
    tb.target_pos(*pos)
    while tb.busy(): tb.update()
    frame, data = tb.get_frame(), tb.req_data()
    time.sleep(0.05)

    full_images.append(frame)
    images.append(cv2.resize(frame, (small_w, small_h)))
    data['x_act'] = 0
    data['y_act'] = 0
    data['z_act'] = 0

    states.append(data)

    tb.press_z(0, 5)
    while tb.busy():
        tb.update()
    pos[2] = tb.req_data()['z']

    while tb.busy():
        tb.update()

    def normalize_pos(pos):
        mX, mY, mZ = 6000, 12000, 1300
        pos[0] = min(mX, max(0, pos[0]))
        pos[1] = min(mY, max(0, pos[1]))
        pos[2] = min(mZ, max(0, pos[2]))

    def millis():
        return int(round(time.time() * 1000))

    act = None
    slip = False
    corr_next = False

    for n in range(num_steps):

        if n % 3 == 0:
            act = policy(pos)
        pos = [pos[i] + act[i] for i in range(3)]
        if corr_next:
            pos[2] -= 15
        normalize_pos(pos)

        tb.target_pos(*pos)
        bt = millis()

        while tb.busy():
            tb.update()

        print(millis() - bt)
        frame, data = tb.get_frame(), tb.req_data()
        data['x_act'] = act[0]
        data['y_act'] = act[1]
        if corr_next:
            data['z_act'] = act[2] - 15
        else:
            data['z_act'] = act[2]

        print(data)
        forces = [data['force_1'], data['force_2'], data['force_3'], data['force_4']]
        avg = sum(forces) / 4

        if avg > max_force:
            corr_next = True
            num_corr += 1
        else:
            corr_next = False

        if (max(forces) < min_force):
            print("Slip detected")
            slip = True

        data['slip'] = slip

        full_images.append(frame)
        images.append(cv2.resize(frame, (small_w, small_h)))

        states.append(data)

    pos[2] = ZERO_POS[2]
    # tb.reset_z()
    # while tb.busy():
    #    tb.update()
    tb.target_pos(*pos)
    while tb.busy():
        tb.update()

    # for i in range(0, len(images), 5):
    #   plt.imshow(images[i])
    #   plt.show()

    print("Corrections: " + str(num_corr))
    return {'images': np.array(images), 'states': np.array(states), 'full_images': np.array(full_images)}


ctimestr = datetime.datetime.now().strftime("%Y-%m-%d:%H:%M:%S")

with open('bearing_stats.pkl', 'rb') as f:
    stats = pkl.load(f)
    mean, std = stats['mean'], stats['std']

for i in range(2000):
    if not i % 100:
        tb.reset()
        while tb.busy():
            tb.update()

    traj = run_traj(18, random_actions)

    time.sleep(3)
    save_traj.save_tf_record('traj_data/' + ctimestr + '/traj' + str(i) + '/', 'traj' + str(i), traj)

tb.reset();
while tb.busy():
    tb.update()
