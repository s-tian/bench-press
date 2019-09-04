import tensorflow as tf
import keras 
from keras.applications import ResNet50
from keras import backend as K
import numpy as np
from scipy.io import loadmat
import cv2
import os 
import tqdm
import time

BATCH_SIZE = 10 

h, w, ch = 270, 360, 3
f_h, f_w, ch = 480, 640, 3
SHAPE_POS = {'octagon': {'x': 2700, 'y': 1000},
             'cube': {'x': 2700, 'y': 3950},
             'pyramid': { 'x': 2700, 'y': 7400},
             'hemisphere': {'x': 2700, 'y': 10350},}
             
def load_data(path, split=(0.8, 0.2)):
    filenames = next(os.walk(path))[2]
    frames = []
    offsets = []
    for mat in filenames:
        print('Loading file: ' + mat)
        data = loadmat(path + mat) 
        frames.extend(data['press_frames'])
        shapes = data['shape'] 
        xs = data['x'][0]
        ys = data['y'][0]

        for i in range(len(data['press_frames'])):
            x_0, y_0 = SHAPE_POS[shapes[i]]['x'], SHAPE_POS[shapes[i]]['y']
            x, y = xs[i], ys[i]
            x_off, y_off = x - x_0, y - y_0
            offsets.append([x_off, y_off])

    # Source for this: Kurt 
    frames = np.array(frames)
    small_frames = []
    print('Resizing images...')
    for frame in tqdm.tqdm(frames):
        small_frames.append(cv2.resize(frame, dsize=(w, h)))
    frames = np.array(small_frames)
    frames = frames / 255.
    offsets = np.array(offsets) 
    data_mean, data_var = np.mean(offsets), np.std(offsets)
    print('Data mean: {}'.format(data_mean))
    print('Data var: {}'.format(data_var))
    offsets = (offsets - np.mean(offsets)) / np.std(offsets)
    permutation = np.random.permutation(len(frames))
    num_holdout = int(frames.shape[0]*split[1])
    x_train, x_test = frames[permutation[num_holdout:]], frames[permutation[:num_holdout]]
    y_train, y_test = offsets[permutation[num_holdout:]], offsets[permutation[:num_holdout]]
    return x_train, x_test, y_train, y_test, data_mean, data_var

def build_model(inp):
    resnet = ResNet50(weights='imagenet', include_top=False, input_tensor=inp, 
                      input_shape=(h, w, ch))
    out = resnet.output
    # Flatten conv layer output
    x = tf.contrib.layers.flatten(out)
    # Feed through 2 FC layers w/ ReLU activation, then get output
    x = tf.layers.dense(x, 512, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer1', activation=tf.nn.relu)
    x = tf.layers.dense(x, 512, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer2', activation=tf.nn.relu)
    x = tf.layers.dense(x, 256, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer3', activation=tf.nn.relu)
    x = tf.layers.dense(x, 256, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer4', activation=tf.nn.relu)
    x = tf.layers.dense(x, 128, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer5', activation=tf.nn.relu)
    x = tf.layers.dense(x, 2, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer6')
    return x 

x_train, x_test, y_train, y_test, data_mean, data_var = load_data('shapes_data/')

X = tf.placeholder("float", shape=[None, h, w, ch])
Y = tf.placeholder("float", shape=[None, 2])

K.set_learning_phase(1)

model = build_model(X)
loss = tf.losses.mean_squared_error(Y, model)
tf.summary.scalar('loss', loss)

with tf.variable_scope("layer1", reuse=True):
    weights1 = tf.get_variable("kernel")
    bias1 = tf.get_variable("bias") 
with tf.variable_scope("layer2", reuse=True):
    weights2 = tf.get_variable("kernel")
    bias2 = tf.get_variable("bias") 
with tf.variable_scope("layer3", reuse=True):
    weights3 = tf.get_variable("kernel")
    bias3 = tf.get_variable("bias") 

#train_operation = tf.train.AdamOptimizer(0.001).minimize(loss, 
#                  var_list=[weights1, weights2, weights3, bias1, bias2, bias3])

train_operation = tf.train.AdamOptimizer().minimize(loss)
inflated_l1loss = tf.reduce_mean(tf.losses.absolute_difference(Y, model, reduction=tf.losses.Reduction.NONE)) * data_var 
tf.summary.scalar('denormalized_absolute_l1_loss', inflated_l1loss)

def update_d(prev, new):
    combined = prev.copy()
    combined.update(new)
    return combined

def train_model(sess, train_X, train_Y, test_X, test_Y, train_operation,
                predict_op, num_epochs, batch_size, test_size,
                train_feed=dict(), test_feed=dict(), howOften = 20):
    losses = []
    startingTime = time.time()
    
    # Total test set acc 
    total_l = 0
    n = 0
    
    K.set_learning_phase(0)
    for start in range(0, len(test_X), batch_size):
        end = start + batch_size
        tX, tY = test_X[start:end], test_Y[start:end]
        l = sess.run(loss, feed_dict = update_d(test_feed, {X: tX, Y: tY}))
        total_l += l
        n += 1

    print("Initial loss was %.04f" % (total_l / n))
        
    with tqdm.tqdm(total=num_epochs*len(train_X)) as ranger:
        i = 0
        for epoch in range(num_epochs):
            for start in range(0, len(train_X), batch_size):
                end = start + batch_size
                K.set_learning_phase(1)
                summary, train = sess.run([summarize, train_operation],
                    feed_dict = update_d(train_feed, {X: train_X[start:end], Y: train_Y[start:end]}))
                train_writer.add_summary(summary, i)
                ranger.update(batch_size)
                i += 1
                if (start//batch_size) % howOften == 0:
                    K.set_learning_phase(0)
                    testSet = np.random.choice(len(test_X), test_size, replace=False)
                    tX, tY = test_X[testSet], test_Y[testSet]
                    summary, l = sess.run([summarize, loss], feed_dict = update_d(test_feed, {X: tX, Y: tY}))
                    test_writer.add_summary(summary, i)
                    losses.append(l)
                    ranger.set_description("Test Loss: " + str(losses[-1]))
                    l = sess.run(predict_op, feed_dict = update_d(test_feed, {X: tX, Y: tY}))
                    print('Renormed loss on avg: {}'.format(l))

            K.set_learning_phase(0)
            testSet = np.random.choice(len(test_X), test_size, replace=False)
            tX, tY = test_X[testSet], test_Y[testSet]
            l = sess.run(loss, feed_dict = update_d(test_feed, {X: tX, Y: tY}))
            losses.append(l)
            ranger.set_description("Test Loss: " + str(losses[-1]))
    
    timeTaken = time.time() - startingTime
    print("Finished training for %d epochs" % num_epochs)
    print("Took %.02f seconds (%.02f s per epoch)" % (timeTaken, timeTaken/num_epochs))
    # Total test set acc 
    total_l = 0
    n = 0
    K.set_learning_phase(0)
    for start in range(0, len(test_X), batch_size):
        end = start + batch_size
        tX, tY = test_X[start:end], test_Y[start:end]
        total_l += sess.run(loss, feed_dict = update_d(test_feed, {X: tX, Y: tY}))
        n += 1
    print("Final loss was %.04f" % (total_l / n))
    #
     
    #accuracies.append(sess.run(accuracy_operation, feed_dict = update_d(test_feed, {X: test_X, Y: test_Y})))
    #print("Final accuracy was %.04f" % accuracies[-1])

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
saver = tf.train.Saver()

summarize = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('logs/train')
test_writer = tf.summary.FileWriter('logs/test')

with tf.Session(config=config) as sess:
    
    tf.global_variables_initializer().run()
    train_model(sess, x_train, y_train, x_test, y_test, train_operation, inflated_l1loss, 100, BATCH_SIZE, 60)
    saver.save(sess, "/tmp/offset_model.ckpt")
