import tensorflow as tf
import numpy as np
import os
import deepdish as dd
import moviepy.editor as mpy
import cv2
import pickle

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def save_tf_record(dir, filename, traj, mean, std):
    if not os.path.exists(dir):
        os.makedirs(dir)
    #images = [ i for i in traj['images']]
    #clip = mpy.ImageSequenceClip(images, fps=25)
    #clip.write_gif(dir + filename + '.gif')
    
    pkl = os.path.join(dir, filename + '.pkl')
    with open(pkl, 'wb') as handle:
        pickle.dump(traj['states'], handle)
    
    for i, im in enumerate(traj['full_images']):
        cv2.imwrite(os.path.join(dir, filename + '_' + str(i) + '.jpg'), im)

    filename = os.path.join(dir, filename + '.tfrecords')
    print(('Writing', filename))
    writer = tf.python_io.TFRecordWriter(filename)

    traj_len = traj['images'].shape[0]

    feature = {}
    for i in range(1, traj_len):

        for feat in traj['states'][i]:
            if feat != 'slip':
                traj['states'][i][feat] = (traj['states'][i][feat] - mean[feat]) / std[feat]

        act = [traj['states'][i]['x_act'], traj['states'][i]['y_act'], traj['states'][i]['z_act']]
        state = [
            traj['states'][i]['x'],
            traj['states'][i]['y'],
            traj['states'][i]['z'],
            traj['states'][i]['slip'],
            traj['states'][i]['force_1'],
            traj['states'][i]['force_2'],
            traj['states'][i]['force_3'],
            traj['states'][i]['force_4']
        ]
        img = traj['images'][i]
        feature['%d/img' % (i - 1)] = _bytes_feature(img.tostring())
        feature['%d/action' % (i - 1)] = _float_feature(act)
        feature['%d/state' % (i - 1)] = _float_feature(state)

    pre_img = traj['images'][0]
    feature['pre_img'] = _bytes_feature(pre_img.tostring())

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())

    writer.close()


def save_dd_record(dir, fname, traj):
    if not os.path.exists(dir):
        os.makedirs(dir)
    filename = os.path.join(dir, fname+ '.hd5')
    images = [ i for i in traj['images']]
    clip = mpy.ImageSequenceClip(images, fps=25)
    clip.write_gif(dir + fname + '.gif')
    side_images = [i for i in traj['side_images']]
    clip2 = mpy.ImageSequenceClip(images, fps=25)
    clip.write_gif(dir + fname + '_side.gif')
    print(('Writing', filename))
    dd.io.save(filename, traj)     
    
