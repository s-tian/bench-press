import tensorflow as tf
import numpy as np
from tqdm import tqdm
import glob
import sys
import os
from scipy.io import loadmat
import argparse

'''
Script to convert data stored in .mat files (collect_data.py) to .tfrecord
'''

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

parser = argparse.ArgumentParser(description='Convert .mat files to .tfrecord')
parser.add_argument('inp_path', metavar='inp_path', type=str, help='directory containing .mat files')
parser.add_argument('out_path', metavar='out_path', type=str, help='directory to output .tfrecord files to. If it does not exist, it wil be created.')
parser.add_argument('-n', '--num', metavar='record_size', type=int, default=5, help='number of examples to store in each .tfrecord file')
parser.add_argument('-p_train', metavar='p_train', type=float, default=0.75, help='proportion of examples, on average, to put in training set')
parser.add_argument('-p_test', metavar='p_test', type=float, default=0.2, help='proportion of examples, on average, to put in test set')
parser.add_argument('-p_val', metavar='p_val', type=float, default=0.05, help='proportion of examples, on average, to put into validation set')

args = parser.parse_args()

data_path = args.inp_path
record_size = args.num

output_dir = args.out_path

# Make dirs if they don't already exist.
dirs = ['', 'train', 'test', 'val']

for d in dirs:
    if not os.path.exists(output_dir + d):
        os.makedirs(output_dir + d)

mat_paths = glob.glob('{}*'.format(data_path))

split = (args.p_train, args.p_test, args.p_val)

assert sum(split) == 1, "Proportions for example distribution don't sum to one."

train = []
test = []
val = []

train_ind = 0
test_ind = 0
val_ind = 0

classes = {
    'square': 0,
    'triangle': 1,
    'star': 2,
    'cylinder': 3,
    'side-cylinder': 4,
    'hemisphere': 5
}


for fname in tqdm(mat_paths):
    
    data = loadmat(fname)
    press_frames = data['press_frames']
    pre_press_frames = data['pre_press_frames']
    x = data['x'][0]
    y = data['y'][0]
    z = data['z'][0]
    thresh = data['force_thresh'][0]
    force1 = data['force_1'][0]
    force2 = data['force_2'][0]
    force3 = data['force_3'][0]
    force4 = data['force_4'][0]
    shapes = [classes[i] for i in data['shape']]
    
    for i in tqdm(range(len(press_frames)), desc='Converting {}...'.format(fname)):
        feature = {
                    'press_img': _bytes_feature(press_frames[i].tostring()),
                    'pre_press_img': _bytes_feature(pre_press_frames[i].tostring()),
                    'x': _int64_feature(x[i].item()),
                    'y': _int64_feature(y[i].item()),
                    'z': _int64_feature(z[i].item()),
                    'thresh': _float_feature(thresh[i].item()),
                    'force1': _float_feature(force1[i]),
                    'force2': _float_feature(force2[i]),
                    'force3': _float_feature(force3[i]),
                    'force4': _float_feature(force4[i]),
                    'shapes': _int64_feature(shapes[i])
                  }
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
            writer = tf.python_io.TFRecordWriter('{}train/train_example_{}_to_{}.tfrecord'.format(output_dir, train_ind, train_ind + record_size - 1))
            for ex in train:
                writer.write(ex.SerializeToString()) 
            train_ind += record_size
            train = []
        if len(test) == record_size:
            writer = tf.python_io.TFRecordWriter('{}test/test_{}_to_{}.tfrecord'.format(output_dir, test_ind, test_ind + record_size - 1))
            for ex in test:
                writer.write(ex.SerializeToString()) 
            test_ind += record_size 
            test = []
        if len(val) == record_size:
            writer = tf.python_io.TFRecordWriter('{}val/_val{}_to_{}.tfrecord'.format(output_dir, val_ind, val_ind + record_size - 1))
            for ex in val:
                writer.write(ex.SerializeToString()) 
            val_ind += record_size 
            val = []
                 
