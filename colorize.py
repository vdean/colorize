import tensorflow as tf
import numpy as np

images = tf.placeholder(tf.float32, [None, 256, 256, 1])

filters1_1 = tf.Variable(tf.truncated_normal([1, 3, 3, 64]))
bias1_1 = tf.Variable(tf.constant(0.1, shape=[256,256,64]))
conv1_1 = tf.relu(tf.nn.conv2d(images, filters1_1, strides=[1,1,1,1], padding='SAME') + bias1_1)

filters1_2 = tf.Variable(tf.truncated_normal([1, 3, 3, 64]))
bias1_2 = tf.Variable(tf.constant(0.1, shape=[128,128,64]))
conv1_2 = tf.relu(tf.nn.conv2d(conv1_1, filters1_2, strides=[2,2,1,1], padding='SAME') + bias1_2)

filters2 = tf.Variable(tf.truncated_normal([1, 3, 3, 64]))
bias2 = tf.Variable(tf.constant(0.1, shape=[256,256,64]))
conv2 = tf.relu(tf.nn.conv2d(conv1_2, filters2, strides=[2,2,1,1], padding='SAME') + bias1)

filters1 = tf.Variable(tf.truncated_normal([1, 3, 3, 64]))
bias1 = tf.Variable(tf.constant(0.1, shape=[256,256,64]))
conv1 = tf.relu(tf.nn.conv2d(images, filters1, strides=[1,1,1,1], padding='SAME') + bias1)

filters1 = tf.Variable(tf.truncated_normal([1, 3, 3, 64]))
bias1 = tf.Variable(tf.constant(0.1, shape=[256,256,64]))
conv1 = tf.relu(tf.nn.conv2d(images, filters1, strides=[1,1,1,1], padding='SAME') + bias1)

filters1 = tf.Variable(tf.truncated_normal([1, 3, 3, 64]))
bias1 = tf.Variable(tf.constant(0.1, shape=[256,256,64]))
conv1 = tf.relu(tf.nn.conv2d(images, filters1, strides=[1,1,1,1], padding='SAME') + bias1)
