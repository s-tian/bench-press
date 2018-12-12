import tensorflow as tf
import numpy as np
from tqdm import tqdm
import glob
import pickle
import os
import cv2
import argparse

'''
Script to convert data stored in .mat files (collect_data.py) to .tfrecord
'''

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value = [value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

parser = argparse.ArgumentParser(description='Convert full trajectory folders to .tfrecord')
parser.add_argument('inp_path', metavar='inp_path', type=str, help='directory containing trajectory subdirectories')
parser.add_argument('out_path', metavar='out_path', type=str, help='directory to output .tfrecord files to. If it does not exist, it wil be created.')
parser.add_argument('-n', '--num', metavar='record_size', type=int, default=1, help='number of examples to store in each .tfrecord file')
parser.add_argument('-p_train', metavar='p_train', type=float, default=0.95, help='proportion of examples, on average, to put in training set')
parser.add_argument('-p_test', metavar='p_test', type=float, default=0.025, help='proportion of examples, on average, to put in test set')
parser.add_argument('-p_val', metavar='p_val', type=float, default=0.025, help='proportion of examples, on average, to put into validation set')

args = parser.parse_args()

data_path = args.inp_path
record_size = args.num

output_dir = args.out_path

# Make dirs if they don't already exist.
dirs = ['', 'train', 'test', 'val']

for d in dirs:
    if not os.path.exists(output_dir + d):
        os.makedirs(output_dir + d)

folders = ['2018-12-05:20:45:28', '2018-12-04:22:25:06', '2018-12-05:19:57:02', '2018-12-05:21:57:26', '2018-12-05:22:02:59']
traj_paths = []
for folder in folders:
    traj_paths.extend(glob.glob('traj_data/{}/traj*/'.format(folder)))

split = (args.p_train, args.p_test, args.p_val)

assert sum(split) == 1, "Proportions for example distribution don't sum to one."

train = []
test = []
val = []

train_ind = 0
test_ind = 0
val_ind = 0

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

pickle_stats_dir = 'dice_stats.pkl'
with open('dice_stats.pkl', 'rb') as f:
    stats = pickle.load(f)
    mean, std = stats['mean'], stats['std']


slip = 0
for fname in tqdm(traj_paths):
    feature = {}
    data = pickle.load(open(glob.glob(fname + '*.pkl')[0], 'rb'))
    if data[1]['slip'] == 1:
        slip += 1 
    start_pos = [data[0]['x'], data[0]['y'], data[0]['z']]

    for i in range(1, len(data)):
        step_data = data[i]
        img = cv2.imread(glob.glob(fname + 'traj*_{}.jpg'.format(i))[0])
        img = cv2.resize(img, dsize=(64, 48))
        act = [step_data['x_act'], step_data['y_act'], step_data['z_act']]
        if i == 9 or i == 18:
            act[2] = data[i-1]['z_act']
            backtrack = [step_data['x'] - start_pos[0], step_data['y'] - start_pos[1], 0]
            act[0] += -backtrack[0]/2
            act[1] += -backtrack[1]/2
        act[0] = (act[0] - mean['x_act']) / std['x_act']
        act[1] = (act[1] - mean['y_act']) / std['y_act']
        if i != 9 and i != 18:
            act[2] = (act[2] - mean['z_act']) / std['z_act']
        for feat in step_data:
            if feat != 'slip':
                step_data[feat] = (step_data[feat] - mean[feat]) / std[feat]
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
        feature['%d/img' % (i-1)] = _bytes_feature(img.tostring())
        feature['%d/action' % (i-1)] = _float_feature(act)
        feature['%d/state' % (i-1)] = _float_feature(state)

    pre_img = cv2.imread(glob.glob(fname + '/traj*_0.jpg')[0])
    pre_img = cv2.resize(pre_img, dsize=(64, 48))
    feature['pre_img'] = _bytes_feature(pre_img.tostring())

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Randomly determine which set to add to

    draw = np.random.rand()

    if draw < split[0]:
        train.append(example)
    elif draw < split[0] + split[1]:
        test.append(example)
    else:
        val.append(example)

    if len(train) == record_size:
        writer = tf.python_io.TFRecordWriter('{}train/train_{}.tfrecord'.format(output_dir, train_ind))
        for ex in train:
            writer.write(ex.SerializeToString())
        train_ind += record_size
        train = []
    if len(test) == record_size:
        writer = tf.python_io.TFRecordWriter('{}test/test_{}.tfrecord'.format(output_dir, test_ind))
        for ex in test:
            writer.write(ex.SerializeToString())
        test_ind += record_size
        test = []
    if len(val) == record_size:
        writer = tf.python_io.TFRecordWriter('{}val/val_{}.tfrecord'.format(output_dir, val_ind))
        for ex in val:
            writer.write(ex.SerializeToString())
        val_ind += record_size
        val = []

# Clear out data in 'incomplete' files

if len(train) > 0:
    writer = tf.python_io.TFRecordWriter('{}train/train_{}.tfrecord'.format(output_dir, train_ind, train_ind + len(train) - 1))
    for ex in train:
        writer.write(ex.SerializeToString())

if len(test) > 0:
    writer = tf.python_io.TFRecordWriter('{}test/test_{}.tfrecord'.format(output_dir, test_ind, test_ind + len(test) - 1))
    for ex in test:
        writer.write(ex.SerializeToString())

if len(val) > 0:
    writer = tf.python_io.TFRecordWriter('{}val/val_{}.tfrecord'.format(output_dir, val_ind, val_ind + len(val) - 1))
    for ex in val:
        writer.write(ex.SerializeToString())

print('Done converting {} tfrec files.'.format(len(traj_paths)))
print(slip)
