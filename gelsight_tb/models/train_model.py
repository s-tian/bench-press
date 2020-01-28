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

BATCH_SIZE = 20 

h, w, ch = 240, 320, 3
f_h, f_w, ch = 480, 640, 3
num_classes = 6
classes = {
    'star': 2,
    'triangle': 1,
    'square': 0,
    'cylinder': 3,
    'side-cylinder': 4,
    'hemisphere': 5
}

def load_data(path, num_holdout):
    filenames = next(os.walk(path))[2]
    frames = []
    labels = []
    for mat in filenames:
        print('Loading file: ' + mat)
        data = loadmat(path + mat) 
        frames.extend(data['press_frames'])
        labels.extend([classes[i] for i in data['shape']])

    # Source for this: Kurt 
    permutation = np.random.permutation(len(frames))
    frames = np.array(frames)
    frames = np.array([cv2.resize(frame, dsize=(w, h)) for frame in frames])
    frames = frames / 255.
    #mean = np.mean(frames, 0)
    #std = np.std(frames, 0)
    #frames -= mean
    #frames /= std
    
    labels = np.array(labels)
    x_train, x_test = frames[permutation[num_holdout:]], frames[permutation[:num_holdout]]
    y_train, y_test = labels[permutation[num_holdout:]], labels[permutation[:num_holdout]]
    return x_train, x_test, y_train, y_test 

def build_model(inp):
    resnet = ResNet50(weights='imagenet', include_top=False, input_tensor=inp, 
                      input_shape=(h, w, ch))
    out = resnet.output
    x = tf.contrib.layers.flatten(out)
    x = tf.layers.dense(x, 64, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer1', activation=tf.nn.relu)
    x = tf.layers.dense(x, 64, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer2', activation=tf.nn.relu)
    x = tf.layers.dense(x, num_classes, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer3')
    return x 

def update_d(prev, new):
    combined = prev.copy()
    combined.update(new)
    return combined

x_train, x_test, y_train, y_test = load_data('good_data/', 500)

X = tf.placeholder("float", shape=[None, h, w, ch])
Y = tf.placeholder("int64", shape=[None])

K.set_learning_phase(1)

model = build_model(X)
one_hot = tf.one_hot(Y, num_classes, axis=-1)
loss = tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=one_hot)

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
predict_operation = tf.argmax(model, 1)
accuracy_operation = tf.reduce_mean(tf.cast(tf.equal(predict_operation, Y), tf.float32))


def train_model(sess, train_X, train_Y, test_X, test_Y, train_operation,
                accuracy_operation, num_epochs, batch_size, test_size,
                train_feed=dict(), test_feed=dict(), howOften = 20):
    accuracies = []
    startingTime = time.time()
    
    # Total test set acc 
    total_acc = 0
    n = 0
    
    K.set_learning_phase(0)
    for start in range(0, len(test_X), batch_size):
        end = start + batch_size
        tX, tY = test_X[start:end], test_Y[start:end]
        acc = sess.run(accuracy_operation, feed_dict = update_d(test_feed, {X: tX, Y: tY}))
        total_acc += acc
        n += 1
    print("Initial accuracy was %.04f" % (total_acc / n))
    #
        
    with tqdm.tqdm(total=num_epochs*len(train_X)) as ranger:
        for epoch in range(num_epochs):
            for start in range(0, len(train_X), batch_size):
                end = start + batch_size
                K.set_learning_phase(1)
                train = sess.run(train_operation,
                    feed_dict = update_d(train_feed, {X: train_X[start:end], Y: train_Y[start:end]}))
                
                ranger.update(batch_size)
                
                if (start//batch_size) % howOften == 0:
                    K.set_learning_phase(0)
                    testSet = np.random.choice(len(test_X), test_size, replace=False)
                    tX, tY = test_X[testSet], test_Y[testSet]
                    acc = sess.run(accuracy_operation, feed_dict = update_d(test_feed, {X: tX, Y: tY}))
                    accuracies.append(acc)
                    ranger.set_description("Test Accuracy: " + str(accuracies[-1]))


            K.set_learning_phase(0)
            testSet = np.random.choice(len(test_X), test_size, replace=False)
            tX, tY = test_X[testSet], test_Y[testSet]
            acc = sess.run(accuracy_operation, feed_dict = update_d(test_feed, {X: tX, Y: tY}))
            accuracies.append(acc)
            ranger.set_description("Test Accuracy: " + str(accuracies[-1]))
    
    timeTaken = time.time() - startingTime
    print("Finished training for %d epochs" % num_epochs)
    print("Took %.02f seconds (%.02f s per epoch)" % (timeTaken, timeTaken/num_epochs))
    # Total test set acc 
    total_acc = 0
    n = 0
    K.set_learning_phase(0)
    for start in range(0, len(test_X), batch_size):
        end = start + batch_size
        tX, tY = test_X[start:end], test_Y[start:end]
        total_acc += sess.run(accuracy_operation, feed_dict = update_d(test_feed, {X: tX, Y: tY}))
        n += 1
    print("Final accuracy was %.04f" % (total_acc / n))
    #
     
    #accuracies.append(sess.run(accuracy_operation, feed_dict = update_d(test_feed, {X: test_X, Y: test_Y})))
    #print("Final accuracy was %.04f" % accuracies[-1])

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3

with tf.Session(config=config) as sess:
    
    tf.initialize_all_variables().run()
    train_model(sess, x_train, y_train, x_test, y_test, train_operation, accuracy_operation, 30, BATCH_SIZE, 30)



