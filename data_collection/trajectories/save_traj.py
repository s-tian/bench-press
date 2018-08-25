import tensorflow as tf
import numpy as np
import os
import deepdish as dd

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def save_tf_record(dir, filename, traj):
    filename = os.path.join(dir, filename + '.tfrecords')
    print(('Writing', filename))
    writer = tf.python_io.TFRecordWriter(filename)
    
    feature = {}
    
    traj_len = traj['images'].shape[0]  
    for i in range(traj_len):
        feature[str(i) + '/img'] = _bytes_feature(traj['images'][i].tostring())         
        for key in traj['states'][i]:
            feature[str(i) + '/' + key] = _float_feature([traj['states'][i][key]]) 
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())

    writer.close()

def save_dd_record(dir, filename, traj)
     
    
