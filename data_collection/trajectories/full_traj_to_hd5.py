import tensorflow as tf
import numpy as np
from tqdm import tqdm
import glob
import pickle
import os
import cv2
import argparse
import deepdish as dd

'''
Script to convert data stored in .mat files (collect_data.py) to .tfrecord
'''

parser = argparse.ArgumentParser(description='Convert full trajectory folders to .tfrecord')
parser.add_argument('inp_path', metavar='inp_path', type=str, help='directory containing trajectory subdirectories')
parser.add_argument('out_path', metavar='out_path', type=str, help='directory to output .tfrecord files to. If it does not exist, it wil be created.')
parser.add_argument('-n', '--num', metavar='record_size', type=int, default=1, help='number of examples to store in each .tfrecord file')


args = parser.parse_args()

data_path = args.inp_path

output_dir = args.out_path

# Make dirs if they don't already exist.
dirs = ['']

for d in dirs:
    if not os.path.exists(output_dir + d):
        os.makedirs(output_dir + d)

traj_paths = glob.glob('{}/2018*/traj*/'.format(data_path))

""" force_1: mean: 5.548266944444444 std: 8.618291543401973 z: mean: 1115.4815277777777 std: 34.865522493962516
x_act: mean: -0.36525 std: 40.55583610822862
force_4: mean: 5.346665555555556 std: 5.871973470396116
y_act: mean: 0.3839166666666667 std: 40.9047147811397
y: mean: 6045.260583333334 std: 212.2458477847846
force_3: mean: 6.150440555555555 std: 7.239953607917641
force_2: mean: 6.4838152777777776 std: 4.568602618451527
z_act: mean: 0.06966666666666667 std: 6.070706704238715
x: mean: 2644.52925 std: 209.59929224857643
"""
mean = {
        'force_1': 5.548266944444444,
        'z': 1115.4815277777777,
        'x_act': 0,
        'force_4': 5.346665555555556,
        'y_act': 0,
        'y': 6045.260583333334,
        'force_3': 6.150440555555555,
        'force_2': 6.4838152777777776,
        'z_act': 0,
        'x': 2644.52925
    }
std = {
        'force_1': 8.618291543401973,
        'z': 34.865522493962516,
        'x_act': 40.55583610822862,
        'force_4': 5.871973470396116,
        'y_act': 40.9047147811397,
        'y': 212.2458477847846,
        'force_3': 7.239953607917641,
        'force_2': 4.568602618451527,
        'z_act': 6.070706704238715,
        'x': 209.59929224857643

    }


f = {
    'force_1': [],
    'force_2':[],
    'force_3':[],
    'force_4':[],
    'x': [],
    'y': [],
    'z': [],
    'x_act': [],
    'y_act': [],
    'z_act': []
}

for fname in []:
    data = pickle.load(open(glob.glob(fname + '*.pkl')[0], 'rb'))

    for step_data in data[1:]:
        for item in step_data:
            if item != 'slip':
                f[item].append(step_data[item])
for item in []:
    if item != 'slip':
        print(item + ': mean: ' + str(np.mean(f[item])) + ' std: ' + str(np.std(f[item])))
slip = 0

j = 0

for fname in tqdm(traj_paths):
    data = pickle.load(open(glob.glob(fname + '*.pkl')[0], 'rb'))
    if data[1]['slip'] == 1:
        slip += 1

    traj = {}
    for i in range(1, len(data)):
        step_data = data[i]
        img = cv2.imread(glob.glob(fname + 'traj*_{}.jpg'.format(i))[0])
        img = cv2.resize(img, dsize=(64, 48))
        for feat in step_data:
            if feat != 'slip':
                step_data[feat] = (step_data[feat] - mean[feat]) / std[feat]
        act = [step_data['x_act'], step_data['y_act'], step_data['z_act']]
        state = [
            step_data['x'],
            step_data['y'],
            step_data['z'],
            step_data['slip'],
            step_data['force_1'],
            step_data['force_2'],
            step_data['force_3'],
            step_data['force_4']
        ]
        traj['img_%d' % (i-1)] = img
        traj['action_%d' % (i-1)] = act
        traj['state_%d' % (i-1)] = state

    pre_img = cv2.imread(glob.glob(fname + '/traj*_0.jpg')[0])
    pre_img = cv2.resize(pre_img, dsize=(64, 48))
    traj['pre_img'] = pre_img
    # Randomly determine which set to add to
    dd.io.save(output_dir + 'traj_{}'.format(j), traj)
    j += 1

print('Done converting {} tfrec files.'.format(len(traj_paths)))
print(slip)
