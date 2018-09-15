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

def save_tf_record(dir, filename, traj):
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
    
    feature = {}
    
    traj_len = traj['images'].shape[0]  
    for i in range(traj_len):
        feature[str(i) + '/img'] = _bytes_feature(traj['images'][i].tostring())         
        for key in traj['states'][i]:
            feature[str(i) + '/' + key] = _float_feature([float(traj['states'][i][key])]) 
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
    print(('Writing', filename))
    dd.io.save(filename, traj)     
    
