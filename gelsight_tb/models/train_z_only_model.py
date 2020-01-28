import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import backend as K
import numpy as np
import ipdb
import pickle as pkl
from scipy.io import loadmat
import cv2
import glob
import os 
import tqdm
import time

BATCH_SIZE = 10 

h, w, ch = 270, 360, 3
f_h, f_w, ch = 480, 640, 3
HOME_POS = {
    'x': 2600,
    'y': 1200,
    'z': 1325,
}
             
def load_data(paths, split=(0.8, 0.2)):
    if type(paths) == str:
        paths = [paths]
    frames = []
    offsets = []

    for path in paths:
        filenames = glob.glob(path + '/*.mat')
        print(filenames)
        for mat in filenames:
            print('Loading file: ' + mat)
            data = loadmat(mat) 
            frames.extend(data['press_frames'])
            zs = data['z'][0]

            for i in range(len(data['press_frames'])):
                z_0 = HOME_POS['z']
                z = zs[i]
                z_off = (z - z_0) * 0.04
                offsets.append([z_off])

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
    x = tf.layers.dense(x, 256, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer2', activation=tf.nn.relu)
    x = tf.layers.dense(x, 256, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer3', activation=tf.nn.relu)
    x = tf.layers.dense(x, 256, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer4', activation=tf.nn.relu)
    x = tf.layers.dense(x, 1, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer5')
    return x 

DATA_DIRS = [
'/home/stephentian/gelsight-tb/data_collection/data/2019-09-05:19:47:21',
'/home/stephentian/gelsight-tb/data_collection/data/2019-09-05:18:31:04'
]

x_train, x_test, y_train, y_test, data_mean, data_var = load_data(DATA_DIRS)

X = tf.placeholder("float", shape=[None, h, w, ch])
Y = tf.placeholder("float", shape=[None, 1])

K.set_learning_phase(1)

model = build_model(X)
loss = tf.losses.mean_squared_error(Y, model) * data_var**2
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
inflated_l1loss = tf.losses.absolute_difference(Y, model, reduction=tf.losses.Reduction.NONE) * data_var 
indiv_mse_loss = (Y - model) ** 2 * data_var **2
tf.summary.scalar('denormalized_absolute_l1_loss', tf.reduce_mean(inflated_l1loss))

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
                    print('Renormed loss on avg: {}'.format(np.mean(np.array(l))))

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
    total_l1 = [] 
    indiv_mse = []
    n = 0

    K.set_learning_phase(0)
    for start in range(0, len(test_X), batch_size):
        end = start + batch_size
        tX, tY = test_X[start:end], test_Y[start:end]
        l1, mse = sess.run([predict_op, indiv_mse_loss], feed_dict = update_d(test_feed, {X: tX, Y: tY}))
        total_l1.extend(l1)
        indiv_mse.extend(mse)
        n += 1
    out = {
        'l1': total_l1,
        'mse': indiv_mse
    }
    with open('logs/' + ctimstr + 'test_losses_zonly.pkl', 'wb') as f:
        pkl.dump(out, f)
    print('overall test l1 loss:{}'.format(np.mean(np.array(total_l1))))
     
    #accuracies.append(sess.run(accuracy_operation, feed_dict = update_d(test_feed, {X: test_X, Y: test_Y})))
    #print("Final accuracy was %.04f" % accuracies[-1])

config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth=True
saver = tf.train.Saver()

summarize = tf.summary.merge_all()
ctimestr = datetime.datetime.now().strftime("%Y-%m-%d:%H:%M:%S")

train_writer = tf.summary.FileWriter('logs/' + ctimestr +'/train')
test_writer = tf.summary.FileWriter('logs/' + ctimestr +'/test')

with tf.Session(config=config) as sess:
    
    tf.global_variables_initializer().run()
    train_model(sess, x_train, y_train, x_test, y_test, train_operation, inflated_l1loss, 1, BATCH_SIZE, 60)
    saver.save(sess, "/tmp/offset_model.ckpt")
