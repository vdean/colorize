import tensorflow as tf
import numpy as np

images = tf.placeholder(tf.float32, [None, 256, 256, 1])

# Conv 1
filters1_1 = tf.Variable(tf.truncated_normal([3, 3, 1, 64]))
bias1_1 = tf.Variable(tf.constant(0.1, shape=[256,256,64]))
conv1_1 = tf.relu(tf.nn.conv2d(images, filters1_1, strides=[1,1,1,1], padding='SAME') + bias1_1)

filters1_2 = tf.Variable(tf.truncated_normal([3, 3, 1, 64]))
bias1_2 = tf.Variable(tf.constant(0.1, shape=[128,128,64]))
conv1_2 = tf.relu(tf.nn.conv2d(conv1_1, filters1_2, strides=[2,2,1,1], padding='SAME') + bias1_2)

# Conv 2
filters2_1 = tf.Variable(tf.truncated_normal([3, 3, 1, 128]))
bias2_1 = tf.Variable(tf.constant(0.1, shape=[128,128,128]))
conv2_1 = tf.relu(tf.nn.conv2d(conv1_2, filters2_1, strides=[1,1,1,1], padding='SAME') + bias2_1)

filters2_2 = tf.Variable(tf.truncated_normal([3, 3, 1, 128]))
bias2_2 = tf.Variable(tf.constant(0.1, shape=[64,64,128]))
conv2_2 = tf.relu(tf.nn.conv2d(conv2_1, filters2_2, strides=[2,2,1,1], padding='SAME') + bias2_2)

# Conv 3
filters3_1 = tf.Variable(tf.truncated_normal([3, 3, 1, 256]))
bias3_1 = tf.Variable(tf.constant(0.1, shape=[64,64,256]))
conv3_1 = tf.relu(tf.nn.conv2d(conv2_2, filters3_1, strides=[1,1,1,1], padding='SAME') + bias3_1)

filters3_2 = tf.Variable(tf.truncated_normal([3, 3, 1, 256]))
bias3_2 = tf.Variable(tf.constant(0.1, shape=[64,64,256]))
conv3_2 = tf.relu(tf.nn.conv2d(conv3_1, filters3_2, strides=[1,1,1,1], padding='SAME') + bias3_2)

filters3_3 = tf.Variable(tf.truncated_normal([3, 3, 1, 256]))
bias3_3 = tf.Variable(tf.constant(0.1, shape=[32,32,256]))
conv3_3 = tf.relu(tf.nn.conv2d(conv3_2, filters3_3, strides=[2,2,1,1], padding='SAME') + bias3_3)

# Conv 4
filters4_1 = tf.Variable(tf.truncated_normal([3, 3, 1, 512]))
bias4_1 = tf.Variable(tf.constant(0.1, shape=[32,32,512]))
conv4_1 = tf.relu(tf.nn.conv2d(conv3_3, filters4_1, strides=[1,1,1,1], padding='SAME') + bias4_1)

filters4_2 = tf.Variable(tf.truncated_normal([3, 3, 1, 512]))
bias4_2 = tf.Variable(tf.constant(0.1, shape=[32,32,512]))
conv4_2 = tf.relu(tf.nn.conv2d(conv4_1, filters4_2, strides=[1,1,1,1], padding='SAME') + bias4_2)

filters4_3 = tf.Variable(tf.truncated_normal([3, 3, 1, 512]))
bias4_3 = tf.Variable(tf.constant(0.1, shape=[32,32,512]))
conv4_3 = tf.relu(tf.nn.conv2d(conv4_2, filters4_3, strides=[1,1,1,1], padding='SAME') + bias4_3)

# Conv 5
filters5_1 = tf.Variable(tf.truncated_normal([3, 3, 1, 512]))
bias5_1 = tf.Variable(tf.constant(0.1, shape=[32,32,512]))
conv5_1 = tf.relu(tf.nn.conv2d(conv4_3, filters5_1, strides=[1,1,1,1], padding='SAME') + bias5_1)

filters5_2 = tf.Variable(tf.truncated_normal([3, 3, 1, 512]))
bias5_2 = tf.Variable(tf.constant(0.1, shape=[32,32,512]))
conv5_2 = tf.relu(tf.nn.conv2d(conv5_1, filters5_2, strides=[1,1,1,1], padding='SAME') + bias5_2)

filters5_3 = tf.Variable(tf.truncated_normal([3, 3, 1, 512]))
bias5_3 = tf.Variable(tf.constant(0.1, shape=[32,32,512]))
conv5_3 = tf.relu(tf.nn.conv2d(conv5_2, filters5_3, strides=[1,1,1,1], padding='SAME') + bias5_3)

# Conv 6
filters6_1 = tf.Variable(tf.truncated_normal([3, 3, 1, 512]))
bias6_1 = tf.Variable(tf.constant(0.1, shape=[32,32,512]))
conv6_1 = tf.relu(tf.nn.conv2d(conv5_3, filters6_1, strides=[1,1,1,1], padding='SAME') + bias6_1)

filters6_2 = tf.Variable(tf.truncated_normal([3, 3, 1, 512]))
bias6_2 = tf.Variable(tf.constant(0.1, shape=[32,32,512]))
conv6_2 = tf.relu(tf.nn.conv2d(conv6_1, filters6_2, strides=[1,1,1,1], padding='SAME') + bias6_2)

filters6_3 = tf.Variable(tf.truncated_normal([3, 3, 1, 512]))
bias6_3 = tf.Variable(tf.constant(0.1, shape=[32,32,512]))
conv6_3 = tf.relu(tf.nn.conv2d(conv6_2, filters6_3, strides=[1,1,1,1], padding='SAME') + bias6_3)

# Conv 7
filters7_1 = tf.Variable(tf.truncated_normal([3, 3, 1, 512]))
bias7_1 = tf.Variable(tf.constant(0.1, shape=[32,32,512]))
conv7_1 = tf.relu(tf.nn.conv2d(conv6_3, filters7_1, strides=[1,1,1,1], padding='SAME') + bias7_1)

filters7_2 = tf.Variable(tf.truncated_normal([3, 3, 1, 512]))
bias7_2 = tf.Variable(tf.constant(0.1, shape=[32,32,512]))
conv7_2 = tf.relu(tf.nn.conv2d(conv7_1, filters7_2, strides=[1,1,1,1], padding='SAME') + bias7_2)

filters7_3 = tf.Variable(tf.truncated_normal([3, 3, 1, 512]))
bias7_3 = tf.Variable(tf.constant(0.1, shape=[32,32,512]))
conv7_3 = tf.relu(tf.nn.conv2d(conv7_2, filters7_3, strides=[1,1,1,1], padding='SAME') + bias7_3)

# Conv 8
filters8_1 = tf.Variable(tf.truncated_normal([4, 4, 1, 256]))
bias8_1 = tf.Variable(tf.constant(0.1, shape=[64,64,256]))
conv8_1 = tf.relu(tf.nn.conv2d(conv7_3, filters8_1, strides=[2,2,1,1], padding='SAME') + bias8_1)

filters8_2 = tf.Variable(tf.truncated_normal([3, 3, 1, 256]))
bias8_2 = tf.Variable(tf.constant(0.1, shape=[64,64,256]))
conv8_2 = tf.relu(tf.nn.conv2d(conv8_1, filters8_2, strides=[1,1,1,1], padding='SAME') + bias8_2)

filters8_3 = tf.Variable(tf.truncated_normal([3, 3, 1, 256]))
bias8_3 = tf.Variable(tf.constant(0.1, shape=[64,64,256]))
conv8_3 = tf.relu(tf.nn.conv2d(conv8_2, filters8_3, strides=[1,1,1,1], padding='SAME') + bias8_3)

# Soft Max
