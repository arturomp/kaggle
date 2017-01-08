from __future__ import print_function
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from collections import defaultdict, Counter
import os
import time

# Read competition data files:
train = pd.read_csv("train.csv")
train = train.as_matrix()
test  = pd.read_csv("test.csv")
test  = test.as_matrix()

file_pattern = "fast_submission_temp"

# Shapes of the training and test sets
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))

### Training set has 42000 rows and 785 columns
### Test set has 28000 rows and 784 columns

# code below from 
# github.com/arturomp/udacity-deep-learning/
# tensorflow.org/tutorials/mnist/pros/

image_size = 28

def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def train_valid(data, train_size, valid_size=0):
  valid_dataset, valid_labels = make_arrays(valid_size, image_size)
  train_dataset, train_labels = make_arrays(train_size, image_size)
    
  # shuffle the data to have random validation and training set
  np.random.shuffle(data)
  if valid_dataset is not None:
    valid_dataset = data[:valid_size, 1:].reshape((-1, image_size, image_size))
    valid_labels  = data[:valid_size, :1].ravel()
              
  train_dataset = data[valid_size:, 1:].reshape((-1, image_size, image_size))
  train_labels  = data[valid_size:, :1].ravel()
    
  return valid_dataset, valid_labels, train_dataset, train_labels

valid_size =  4000
train_size = len(train) - valid_size
# test_size  = 28000

valid_dataset, valid_labels, train_dataset, train_labels = train_valid(
  train, train_size, valid_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test.shape)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

num_labels = 10
num_channels = 1 # grayscale

import numpy as np

# map labels to one-hot encoding
def reformat(dataset, labels=None):
  dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
  if labels is not None:
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset , _            = reformat(test)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test.shape)


# Multilayer Convolutional Network

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

keep_rate = 0.5
batch_size = 50
patch_size = 5
depth = 32
num_hidden = 1024
learning_rate = 1e-4
num_steps = 20001


for learning_rate in [1e-3, 1e-4]:
    for keep_rate in [1, 0.95, 0.9]:

        graph = tf.Graph()

        with graph.as_default():

            # x = tf.placeholder(tf.float32, shape=[None, image_size*image_size])
            x = x_image = tf.placeholder(tf.float32, shape=[None,image_size, image_size, num_channels]) 
            # y_ = tf.placeholder(tf.float32, shape=[None, num_labels])
            y_ = tf.placeholder(tf.float32, shape=[None, num_labels])


            W_conv1 = weight_variable([patch_size, patch_size, num_channels, depth])
            b_conv1 = bias_variable([depth])

            # x_image = tf.reshape(x, [-1,image_size, image_size, num_channels])


            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)


            W_conv2 = weight_variable([patch_size, patch_size, depth, depth*2])
            b_conv2 = bias_variable([depth*2])

            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = max_pool_2x2(h_conv2)


            W_fc1 = weight_variable([image_size // 4 * image_size // 4 * depth*2, num_hidden])
            b_fc1 = bias_variable([num_hidden])

            h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


            keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


            W_fc2 = weight_variable([num_hidden, num_labels])
            b_fc2 = bias_variable([num_labels])

            y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
            correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



        with tf.Session(graph=graph) as session:
            tf.initialize_all_variables().run()
            t = time.time()
            for i in range(num_steps):

              batch = [None]*2

              offset = (i * batch_size) % (train_labels.shape[0] - batch_size)
              batch[0] = train_dataset[offset:(offset + batch_size), :, :, :]
              batch[1] = train_labels[offset:(offset + batch_size), :]

              if i%5000 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x:batch[0], y_: batch[1], keep_prob: 1.0})
                print("step %d, training accuracy %g"%(i, train_accuracy))

              if (i % 20000 == 0 and i > 0):
                  acc = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
                  print('{:.2f}%  lr: {:g}, kr: {:g}'.format(acc*100, learning_rate, keep_rate))
              train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: keep_rate})

            
            print("Train time: {:.3f} sec".format(time.time() - t))

            kaggle_pred = y_conv.eval(feed_dict={x:test_dataset, keep_prob: 1.0})


            p_filename = os.path.splitext(os.path.basename(__file__))[0]+"_"+str(learning_rate)+"_"+str(keep_rate)+"_"+str(acc)+"_"+file_pattern+".csv"

            with open(p_filename, "w") as f:
              f.write("ImageId,Label\n")
              for img_id, pred in enumerate(kaggle_pred):
                f.write("{:d},{:d}\n".format(img_id+1, np.argmax(pred)))


# compile results

files = [f for f in os.listdir(os.getcwd()) if file_pattern+".csv" in f]

id_dict = defaultdict(Counter)

# count all results
for f in files:

    ids  = pd.read_csv(f)
    ids  = ids.as_matrix()

    for i, num in ids:
        id_dict[i][num] += 1

# choose the most common classifications
kaggle_pred = []
for k in sorted(id_dict.keys()):
    if len(id_dict[k].keys()) > 1:
        if len(set(id_dict[k].values()))==1: # all digits have the same number of votes
            # kaggle_pred.append(id_dict[k].most_common()[0][0]) # get the first tied result
            kaggle_pred.append(id_dict[k].most_common()[1][0]) # get the second tied result
            # kaggle_pred.append(id_dict[k].most_common()[-1][0]) # get the last tied result
        else:
            kaggle_pred.append(id_dict[k].most_common()[0][0]) # get the most common
    else:
        kaggle_pred.append(id_dict[k].keys()[0])


p_filename = os.path.splitext(os.path.basename(__file__))[0]+"_"+file_pattern+"_compiled.csv"
with open(p_filename, "w") as f:
  f.write("ImageId,Label\n")
  for img_id, pred in enumerate(kaggle_pred):
    f.write("{:d},{:d}\n".format(img_id+1, pred))
