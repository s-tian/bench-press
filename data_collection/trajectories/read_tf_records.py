import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
import glob

data_path = 'gelsight_data/train/*.tfrecord'

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3

data_paths = glob.glob(data_path)
print(data_paths)

with tf.Session(config=config) as sess:
    feature = {'1/img': tf.FixedLenFeature([], tf.string),
               '1/state': tf.FixedLenFeature([8], tf.float32)}
 
    filename_queue = tf.train.string_input_producer(data_paths, num_epochs = 2)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
     
    features = tf.parse_single_example(serialized_example, features=feature)

    image = tf.decode_raw(features['1/img'], tf.uint8)
    state = tf.cast(features['1/state'], tf.float32)

    image = tf.reshape(image, [48, 64, 3])
         

    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(10):
        img = sess.run(image)
        img = img.astype(np.uint8)
     
        plt.imshow(img)
        plt.show()
        state_val = sess.run(state)
        print(state_val)

    coord.request_stop()

    coord.join(threads)

    sess.close()
     
 
