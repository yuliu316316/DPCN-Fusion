# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import scipy.misc
import time
import os
import glob
import cv2

def imread(path):
    return scipy.misc.imread(path, mode='RGB').astype(np.float)
def imsave(image, path):
  return scipy.misc.imsave(path, image)
def rgb2yuv(rgb):
    rgb = rgb.astype(np.float32)

    m = [[0.299, 0.587, 0.114], [-0.147, -0.289, 0.436], [0.615, -0.515, -0.100]]
    shape1 = rgb.shape
    yuv = np.empty(shape1, dtype=np.float32)
    for i in range(3):
        yuv[:, :, i] = rgb[:, :, 0]*m[i][0] + rgb[:, :, 1]*m[i][1] + rgb[:, :, 2]*m[i][2]
    return yuv

def yuv2rgb(yuv):

    mtxYUVtoRGB = np.array([[1.0000, -0.0000, 1.1398],
                            [1.0000, -0.3946, -0.5805],
                            [1.0000,  2.0320, -0.0005]])
    rgb = np.zeros(yuv.shape)
    for i in range(3):
        rgb[:, :, i] = yuv[:, :, 0] * mtxYUVtoRGB[i, 0] + yuv[:, :, 1] * mtxYUVtoRGB[i, 1] + yuv[:, :, 2] * mtxYUVtoRGB[i, 2]
    return rgb
def prepare_data(dataset):
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
    data = glob.glob(os.path.join(data_dir, "*.jpg"))
    data.extend(glob.glob(os.path.join(data_dir, "*.bmp")))
    data.sort(key=lambda x:x[len(data_dir)+1:-4])
    return data

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

def fusion_model(GFP_input, PCI_input ):
    with tf.variable_scope('fusion_model'):
        with tf.variable_scope('layer_G1'):
            weights = tf.get_variable("w_G1", initializer=tf.constant(reader.get_tensor('fusion_model/layer_G1/w_G1')))
            bias = tf.get_variable("b_G1", initializer=tf.constant(reader.get_tensor(('fusion_model/layer_G1/b_G1'))))
            conv_G1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(GFP_input, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv_G1 = lrelu(conv_G1)
        with tf.variable_scope('layer_P1'):
            weights = tf.get_variable("w_P1", initializer=tf.constant(reader.get_tensor('fusion_model/layer_P1/w_P1')))
            bias = tf.get_variable("b_P1", initializer=tf.constant(reader.get_tensor(('fusion_model/layer_P1/b_P1'))))
            conv_P1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(PCI_input, weights, strides=[1, 1, 1, 1], padding='SAME') + bias, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            conv_P1 = lrelu(conv_P1)
        with tf.variable_scope('layer_G2'):
            weights1 = tf.get_variable("wG2_1", initializer=tf.constant(reader.get_tensor('fusion_model/layer_G2/wG2_1')))
            bias1 = tf.get_variable("bG2_1", initializer=tf.constant(reader.get_tensor(('fusion_model/layer_G2/bG2_1'))))
            convG2_1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv_G1, weights1, strides=[1, 1, 1, 1], padding='SAME') + bias1, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            convG2_1 = lrelu(convG2_1)
            # p
            weights2 = tf.get_variable("wG2_2", initializer=tf.constant(reader.get_tensor('fusion_model/layer_G2/wG2_2')))
            bias2 = tf.get_variable("bG2_2", initializer=tf.constant(reader.get_tensor(('fusion_model/layer_G2/bG2_2'))))
            convG2_2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(convG2_1, weights2, strides=[1, 1, 1, 1], padding='SAME') + bias2, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            G2_1 = conv_G1 + convG2_2
            convGP_1 = lrelu(G2_1)
            # p
            weights3 = tf.get_variable("wG2_3", initializer=tf.constant(reader.get_tensor('fusion_model/layer_G2/wG2_3')))
            bias3 = tf.get_variable("bG2_3",initializer=tf.constant(reader.get_tensor(('fusion_model/layer_G2/bG2_3'))))
            convG2_3 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv_P1, weights3, strides=[1, 1, 1, 1], padding='SAME') + bias3, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            convG2_3 = lrelu(convG2_3)
            weights4 = tf.get_variable("wG2_4",initializer=tf.constant(reader.get_tensor('fusion_model/layer_G2/wG2_4')))
            bias4 = tf.get_variable("bG2_4", initializer=tf.constant(reader.get_tensor(('fusion_model/layer_G2/bG2_4'))))
            convG2_4 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(convG2_3, weights4, strides=[1, 1, 1, 1], padding='SAME') + bias4, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            G2_1 = conv_P1 + convG2_4
            convPG_1 = lrelu(G2_1)
            convG2 = tf.concat([convGP_1, convPG_1], axis=-1)
            weights5 = tf.get_variable("wG2_5", initializer=tf.constant(reader.get_tensor('fusion_model/layer_G2/wG2_5')))
            bias5 = tf.get_variable("bG2_5", initializer=tf.constant(reader.get_tensor(('fusion_model/layer_G2/bG2_5'))))
            convG2_5 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(convG2, weights5, strides=[1, 1, 1, 1], padding='VALID') + bias5, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            GP1 = conv_G1 + convG2_5
            convG2 = lrelu(GP1)
        with tf.variable_scope('layer_P2'):
            weights1 = tf.get_variable("wP2_1", initializer=tf.constant(reader.get_tensor('fusion_model/layer_P2/wP2_1')))
            bias1 = tf.get_variable("bP2_1",  initializer=tf.constant(reader.get_tensor(('fusion_model/layer_P2/bP2_1'))))
            convP2_1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv_P1, weights1, strides=[1, 1, 1, 1], padding='SAME') + bias1, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            convP2_1 = lrelu(convP2_1)
            # p
            weights2 = tf.get_variable("wP2_2", initializer=tf.constant(reader.get_tensor('fusion_model/layer_P2/wP2_2')))
            bias2 = tf.get_variable("bP2_2", initializer=tf.constant(reader.get_tensor(('fusion_model/layer_P2/bP2_2'))))
            convP2_2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(convP2_1, weights2, strides=[1, 1, 1, 1], padding='SAME') + bias2, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            P2_1 = conv_P1 + convP2_2
            convPG_1 = lrelu(P2_1)
            # p
            weights3 = tf.get_variable("wP2_3", initializer=tf.constant(reader.get_tensor('fusion_model/layer_P2/wP2_3')))
            bias3 = tf.get_variable("bP2_3",  initializer=tf.constant(reader.get_tensor(('fusion_model/layer_P2/bP2_3'))))
            convP2_3 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(conv_G1, weights3, strides=[1, 1, 1, 1], padding='SAME') + bias3, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            convP2_3 = lrelu(convP2_3)
            weights4 = tf.get_variable("wP2_4",  initializer=tf.constant(reader.get_tensor('fusion_model/layer_P2/wP2_4')))
            bias4 = tf.get_variable("bP2_4",initializer=tf.constant(reader.get_tensor(('fusion_model/layer_P2/bP2_4'))))
            convP2_4 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(convP2_3, weights4, strides=[1, 1, 1, 1], padding='SAME') + bias4, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            P2_1 = conv_G1 + convP2_4
            convGP_1 = lrelu(P2_1)
            convP2 = tf.concat([convGP_1, convPG_1], axis=-1)
            weights5 = tf.get_variable("wP2_5",  initializer=tf.constant(reader.get_tensor('fusion_model/layer_P2/wP2_5')))
            bias5 = tf.get_variable("bP2_5", initializer=tf.constant(reader.get_tensor(('fusion_model/layer_P2/bP2_5'))))
            convP2_5 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(convP2, weights5, strides=[1, 1, 1, 1], padding='VALID') + bias5, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            PG1 = conv_P1 + convP2_5
            convP2 = lrelu(PG1)
        with tf.variable_scope('layer_G3'):
            weights1 = tf.get_variable("wG3_1",  initializer=tf.constant(reader.get_tensor('fusion_model/layer_G3/wG3_1')))
            bias1 = tf.get_variable("bG3_1",  initializer=tf.constant(reader.get_tensor(('fusion_model/layer_G3/bG3_1'))))
            convG3_1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(convG2, weights1, strides=[1, 1, 1, 1], padding='SAME') + bias1, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            convG3_1 = lrelu(convG3_1)
            # p
            weights2 = tf.get_variable("wG3_2",  initializer=tf.constant(reader.get_tensor('fusion_model/layer_G3/wG3_2')))
            bias2 = tf.get_variable("bG3_2",  initializer=tf.constant(reader.get_tensor(('fusion_model/layer_G3/bG3_2'))))
            convG3_2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(convG3_1, weights2, strides=[1, 1, 1, 1], padding='SAME') + bias2, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            G3_1 = convG2 + convG3_2
            convGP_2 = lrelu(G3_1)
            # p
            weights3 = tf.get_variable("wG3_3",  initializer=tf.constant(reader.get_tensor('fusion_model/layer_G3/wG3_3')))
            bias3 = tf.get_variable("bG3_3",initializer=tf.constant(reader.get_tensor(('fusion_model/layer_G3/bG3_3'))))
            convG3_3 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(convP2, weights3, strides=[1, 1, 1, 1], padding='SAME') + bias3, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            convG3_3 = lrelu(convG3_3)
            weights4 = tf.get_variable("wG3_4", initializer=tf.constant(reader.get_tensor('fusion_model/layer_G3/wG3_4')))
            bias4 = tf.get_variable("bG3_4", initializer=tf.constant(reader.get_tensor(('fusion_model/layer_G3/bG3_4'))))
            convG3_4 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(convG3_3, weights4, strides=[1, 1, 1, 1], padding='SAME') + bias4, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            G3_1 = convP2 + convG3_4
            convPG_2 = lrelu(G3_1)
            convG3 = tf.concat([convGP_2, convPG_2], axis=-1)
            weights5 = tf.get_variable("wG3_5", initializer=tf.constant(reader.get_tensor('fusion_model/layer_G3/wG3_5')))
            bias5 = tf.get_variable("bG3_5", initializer=tf.constant(reader.get_tensor(('fusion_model/layer_G3/bG3_5'))))
            convG3_5 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(convG3, weights5, strides=[1, 1, 1, 1], padding='VALID') + bias5, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            GP2 = convG2 + convG3_5
            convG3 = lrelu(GP2)
        with tf.variable_scope('layer_P3'):
            weights1 = tf.get_variable("wP3_1", initializer=tf.constant(reader.get_tensor('fusion_model/layer_P3/wP3_1')))
            bias1 = tf.get_variable("bP3_1", initializer=tf.constant(reader.get_tensor(('fusion_model/layer_P3/bP3_1'))))
            convP3_1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(convP2, weights1, strides=[1, 1, 1, 1], padding='SAME') + bias1, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            convP3_1 = lrelu(convP3_1)
            # p
            weights2 = tf.get_variable("wP3_2",  initializer=tf.constant(reader.get_tensor('fusion_model/layer_P3/wP3_2')))
            bias2 = tf.get_variable("bP3_2", initializer=tf.constant(reader.get_tensor(('fusion_model/layer_P3/bP3_2'))))
            convP3_2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(convP3_1, weights2, strides=[1, 1, 1, 1], padding='SAME') + bias2, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            P3_1 = convP2 + convP3_2
            convPG_3 = lrelu(P3_1)
            # p
            weights3 = tf.get_variable("wP3_3", initializer=tf.constant(reader.get_tensor('fusion_model/layer_P3/wP3_3')))
            bias3 = tf.get_variable("bP3_3", initializer=tf.constant(reader.get_tensor(('fusion_model/layer_P3/bP3_3'))))
            convP3_3 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(convG2, weights3, strides=[1, 1, 1, 1], padding='SAME') + bias3, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            convP3_3 = lrelu(convP3_3)
            weights4 = tf.get_variable("wP3_4",initializer=tf.constant(reader.get_tensor('fusion_model/layer_P3/wP3_4')))
            bias4 = tf.get_variable("bP3_4", initializer=tf.constant(reader.get_tensor(('fusion_model/layer_P3/bP3_4'))))
            convP3_4 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(convP3_3, weights4, strides=[1, 1, 1, 1], padding='SAME') + bias4, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            P3_1 = convG2 + convP3_4
            convGP_3 = lrelu(P3_1)
            convP3 = tf.concat([convGP_3, convPG_3], axis=-1)
            weights5 = tf.get_variable("wP3_5", initializer=tf.constant(reader.get_tensor('fusion_model/layer_P3/wP3_5')))
            bias5 = tf.get_variable("bP3_5",initializer=tf.constant(reader.get_tensor(('fusion_model/layer_P3/bP3_5'))))
            convP3_5 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(convP3, weights5, strides=[1, 1, 1, 1], padding='VALID') + bias5, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            PG3 = convP2 + convP3_5
            convP3 = lrelu(PG3)
        with tf.variable_scope('layer_G4'):
            weights1 = tf.get_variable("wG4_1",  initializer=tf.constant(reader.get_tensor('fusion_model/layer_G4/wG4_1')))
            bias1 = tf.get_variable("bG4_1",initializer=tf.constant(reader.get_tensor(('fusion_model/layer_G4/bG4_1'))))
            convG4_1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(convG3, weights1, strides=[1, 1, 1, 1], padding='SAME') + bias1, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            convG4_1 = lrelu(convG4_1)
            # p
            weights2 = tf.get_variable("wG4_2", initializer=tf.constant(reader.get_tensor('fusion_model/layer_G4/wG4_2')))
            bias2 = tf.get_variable("bG4_2", initializer=tf.constant(reader.get_tensor(('fusion_model/layer_G4/bG4_2'))))
            convG4_2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(convG4_1, weights2, strides=[1, 1, 1, 1], padding='SAME') + bias2, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            G4_1 = convG3 + convG4_2
            convGP_4 = lrelu(G4_1)
            # p
            weights3 = tf.get_variable("wG4_3", initializer=tf.constant(reader.get_tensor('fusion_model/layer_G4/wG4_3')))
            bias3 = tf.get_variable("bG4_3", initializer=tf.constant(reader.get_tensor(('fusion_model/layer_G4/bG4_3'))))
            convG4_3 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(convP3, weights3, strides=[1, 1, 1, 1], padding='SAME') + bias3, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            convG4_3 = lrelu(convG4_3)
            weights4 = tf.get_variable("wG4_4", initializer=tf.constant(reader.get_tensor('fusion_model/layer_G4/wG4_4')))
            bias4 = tf.get_variable("bG4_4",initializer=tf.constant(reader.get_tensor(('fusion_model/layer_G4/bG4_4'))))
            convG4_4 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(convG4_3, weights4, strides=[1, 1, 1, 1], padding='SAME') + bias4, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            G4_1 = convP3 + convG4_4
            convPG_4 = lrelu(G4_1)
            convG4 = tf.concat([convGP_4, convPG_4], axis=-1)
            weights5 = tf.get_variable("wG4_5", initializer=tf.constant(reader.get_tensor('fusion_model/layer_G4/wG4_5')))
            bias5 = tf.get_variable("bG4_5", initializer=tf.constant(reader.get_tensor(('fusion_model/layer_G4/bG4_5'))))
            convG4_5 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(convG4, weights5, strides=[1, 1, 1, 1], padding='VALID') + bias5, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            GP4 = convG3 + convG4_5
            convG4 = lrelu(GP4)
        with tf.variable_scope('layer_P4'):
            weights1 = tf.get_variable("wP4_1", initializer=tf.constant(reader.get_tensor('fusion_model/layer_P4/wP4_1')))
            bias1 = tf.get_variable("bP4_1",initializer=tf.constant(reader.get_tensor(('fusion_model/layer_P4/bP4_1'))))
            convP4_1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(convP3, weights1, strides=[1, 1, 1, 1], padding='SAME') + bias1, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            convP4_1 = lrelu(convP4_1)
            # p
            weights2 = tf.get_variable("wP4_2",initializer=tf.constant(reader.get_tensor('fusion_model/layer_P4/wP4_2')))
            bias2 = tf.get_variable("bP4_2", initializer=tf.constant(reader.get_tensor(('fusion_model/layer_P4/bP4_2'))))
            convP4_2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(convP4_1, weights2, strides=[1, 1, 1, 1], padding='SAME') + bias2, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            P4_1 = convP3 + convP4_2
            convPG_4 = lrelu(P4_1)
            # p
            weights3 = tf.get_variable("wP4_3",initializer=tf.constant(reader.get_tensor('fusion_model/layer_P4/wP4_3')))
            bias3 = tf.get_variable("bP4_3",initializer=tf.constant(reader.get_tensor(('fusion_model/layer_P4/bP4_3'))))
            convP4_3 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(convG3, weights3, strides=[1, 1, 1, 1], padding='SAME') + bias3, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            convP4_3 = lrelu(convP4_3)
            weights4 = tf.get_variable("wP4_4", initializer=tf.constant(reader.get_tensor('fusion_model/layer_P4/wP4_4')))
            bias4 = tf.get_variable("bP4_4", initializer=tf.constant(reader.get_tensor(('fusion_model/layer_P4/bP4_4'))))
            convP4_4 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(convP4_3, weights4, strides=[1, 1, 1, 1], padding='SAME') + bias4, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            P4_1 = convG3 + convP4_4
            convGP_4 = lrelu(P4_1)
            convP4 = tf.concat([convGP_4, convPG_4], axis=-1)
            weights5 = tf.get_variable("wP4_5", initializer=tf.constant(reader.get_tensor('fusion_model/layer_P4/wP4_5')))
            bias5 = tf.get_variable("bP4_5",initializer=tf.constant(reader.get_tensor(('fusion_model/layer_P4/bP4_5'))))
            convP4_5 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(convP4, weights5, strides=[1, 1, 1, 1], padding='VALID') + bias5, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            PG4 = convP3 + convP4_5
            convP4 = lrelu(PG4)
        in4 = tf.concat([convG4, convP4], axis=-1)
        with tf.variable_scope('layer5'):
            #TOP
            weights1 = tf.get_variable("w5_1",  initializer=tf.constant(reader.get_tensor('fusion_model/layer5/w5_1')))
            bias1 = tf.get_variable("b5-1", initializer=tf.constant(reader.get_tensor(('fusion_model/layer5/b5-1'))))
            CONV5t = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(in4, weights1, strides=[1, 1, 1, 1], padding='SAME') + bias1, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            CONV5t = lrelu(CONV5t)
            #LEFT
            weights2 = tf.get_variable("w5_2", initializer=tf.constant(reader.get_tensor('fusion_model/layer5/w5_2')))
            bias2 = tf.get_variable("b5-2", initializer=tf.constant(reader.get_tensor(('fusion_model/layer5/b5-2'))))
            CONV5L = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV5t , weights2, strides=[1, 1, 1, 1], padding='VALID') + bias2, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            #MEDIUM
            #1
            weights3 = tf.get_variable("w5_3", initializer=tf.constant(reader.get_tensor('fusion_model/layer5/w5_3')))
            bias3 = tf.get_variable("b5-3", initializer=tf.constant(reader.get_tensor(('fusion_model/layer5/b5-3'))))
            CONV5M1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV5t, weights3, strides=[1, 1, 1, 1], padding='VALID') + bias3, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            CONV5M1 =lrelu(CONV5M1)
            #2
            weights4 = tf.get_variable("w5_4", initializer=tf.constant(reader.get_tensor('fusion_model/layer5/w5_4')))
            bias4 = tf.get_variable("b5-4", initializer=tf.constant(reader.get_tensor(('fusion_model/layer5/b5-4'))))
            CONV5M2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV5M1, weights4, strides=[1, 1, 1, 1], padding='SAME') + bias4, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            #RIGHT
            #1
            weights5 = tf.get_variable("w5_5",initializer=tf.constant(reader.get_tensor('fusion_model/layer5/w5_5')))
            bias5 = tf.get_variable("b5-5",initializer=tf.constant(reader.get_tensor(('fusion_model/layer5/b5-5'))))
            CONV5R1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV5t, weights5, strides=[1, 1, 1, 1], padding='VALID') + bias5, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            CONV5R1 = lrelu(CONV5R1)
            # 2
            weights6 = tf.get_variable("w5_6", initializer=tf.constant(reader.get_tensor('fusion_model/layer5/w5_6')))
            bias6 = tf.get_variable("b5-6", initializer=tf.constant(reader.get_tensor(('fusion_model/layer5/b5-6'))))
            CONV5R2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV5R1, weights6, strides=[1, 1, 1, 1], padding='SAME') + bias6, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            CONV5R2 = lrelu(CONV5R2)
            #3
            weights7 = tf.get_variable("w5_7",initializer=tf.constant(reader.get_tensor('fusion_model/layer5/w5_7')))
            bias7 = tf.get_variable("b5-7", initializer=tf.constant(reader.get_tensor(('fusion_model/layer5/b5-7'))))
            CONV5R3 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV5R2, weights7, strides=[1, 1, 1, 1], padding='SAME') + bias7, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            OUT5=tf.concat([CONV5L,CONV5M2,CONV5R3 ],axis=-1)
            #bottom
            weights8 = tf.get_variable("w5_8", initializer=tf.constant(reader.get_tensor('fusion_model/layer5/w5_8')))
            bias8 = tf.get_variable("b5-8", initializer=tf.constant(reader.get_tensor(('fusion_model/layer5/b5-8'))))
            convb = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(OUT5, weights8, strides=[1, 1, 1, 1], padding='SAME') + bias8, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            convb=convb+CONV5t
            convb= lrelu( convb )
        with tf.variable_scope('layer6'):
            #TOP
            weights1 = tf.get_variable("w6_1",  initializer=tf.constant(reader.get_tensor('fusion_model/layer6/w6_1')))
            bias1 = tf.get_variable("b6-1", initializer=tf.constant(reader.get_tensor(('fusion_model/layer6/b6-1'))))
            CONV6t = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(convb, weights1, strides=[1, 1, 1, 1], padding='SAME') + bias1, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            CONV6t = lrelu(CONV6t)
            #LEFT
            weights2 = tf.get_variable("w6_2", initializer=tf.constant(reader.get_tensor('fusion_model/layer6/w6_2')))
            bias2 = tf.get_variable("b6-2", initializer=tf.constant(reader.get_tensor(('fusion_model/layer6/b6-2'))))
            CONV6L = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV6t , weights2, strides=[1, 1, 1, 1], padding='VALID') + bias2, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            #MEDIUM
            #1
            weights3 = tf.get_variable("w6_3", initializer=tf.constant(reader.get_tensor('fusion_model/layer6/w6_3')))
            bias3 = tf.get_variable("b6-3", initializer=tf.constant(reader.get_tensor(('fusion_model/layer6/b6-3'))))
            CONV6M1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV6t, weights3, strides=[1, 1, 1, 1], padding='VALID') + bias3, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            CONV6M1 =lrelu(CONV6M1)
            #2
            weights4 = tf.get_variable("w6_4", initializer=tf.constant(reader.get_tensor('fusion_model/layer6/w6_4')))
            bias4 = tf.get_variable("b6-4", initializer=tf.constant(reader.get_tensor(('fusion_model/layer6/b6-4'))))
            CONV6M2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV6M1, weights4, strides=[1, 1, 1, 1], padding='SAME') + bias4, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            #RIGHT
            #1
            weights5 = tf.get_variable("w6_5",initializer=tf.constant(reader.get_tensor('fusion_model/layer6/w6_5')))
            bias5 = tf.get_variable("b6-5",initializer=tf.constant(reader.get_tensor(('fusion_model/layer6/b6-5'))))
            CONV6R1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV6t, weights5, strides=[1, 1, 1, 1], padding='VALID') + bias5, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            CONV6R1 = lrelu(CONV6R1)
            # 2
            weights6 = tf.get_variable("w6_6", initializer=tf.constant(reader.get_tensor('fusion_model/layer6/w6_6')))
            bias6 = tf.get_variable("b6-6", initializer=tf.constant(reader.get_tensor(('fusion_model/layer6/b6-6'))))
            CONV6R2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV6R1, weights6, strides=[1, 1, 1, 1], padding='SAME') + bias6, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            CONV6R2 = lrelu(CONV6R2)
            #3
            weights7 = tf.get_variable("w6_7",initializer=tf.constant(reader.get_tensor('fusion_model/layer6/w6_7')))
            bias7 = tf.get_variable("b6-7", initializer=tf.constant(reader.get_tensor(('fusion_model/layer6/b6-7'))))
            CONV6R3 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV6R2, weights7, strides=[1, 1, 1, 1], padding='SAME') + bias7, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            OUT6=tf.concat([CONV6L,CONV6M2,CONV6R3 ],axis=-1)
            #bottom
            weights8 = tf.get_variable("w6_8", initializer=tf.constant(reader.get_tensor('fusion_model/layer6/w6_8')))
            bias8 = tf.get_variable("b6-8", initializer=tf.constant(reader.get_tensor(('fusion_model/layer6/b6-8'))))
            convb6 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(  OUT6, weights8, strides=[1, 1, 1, 1], padding='SAME') + bias8, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            convb6 =convb6 +CONV6t
            convb6= lrelu(convb6)
        l7 = in4+convb6
        with tf.variable_scope('layer8'):
            # TOP
            weights1 = tf.get_variable("w8_1",  initializer=tf.constant(reader.get_tensor('fusion_model/layer8/w8_1')))
            bias1 = tf.get_variable("b8-1", initializer=tf.constant(reader.get_tensor(('fusion_model/layer8/b8-1'))))
            CONV8t = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(l7, weights1, strides=[1, 1, 1, 1], padding='SAME') + bias1, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            CONV8t = lrelu(CONV8t)
            # LEFT
            weights2 = tf.get_variable("w8_2",  initializer=tf.constant(reader.get_tensor('fusion_model/layer8/w8_2')))
            bias2 = tf.get_variable("b8-2", initializer=tf.constant(reader.get_tensor(('fusion_model/layer8/b8-2'))))
            CONV8L = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV8t, weights2, strides=[1, 1, 1, 1], padding='VALID') + bias2, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            # MEDIUM
            # 1
            weights3 = tf.get_variable("w8_3", initializer=tf.constant(reader.get_tensor('fusion_model/layer8/w8_3')))
            bias3 = tf.get_variable("b8-3",initializer=tf.constant(reader.get_tensor(('fusion_model/layer8/b8-3'))))
            CONV8M1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV8t, weights3, strides=[1, 1, 1, 1], padding='VALID') + bias3, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            CONV8M1 = lrelu(CONV8M1)
            # 2
            weights4 = tf.get_variable("w8_4", initializer=tf.constant(reader.get_tensor('fusion_model/layer8/w8_4')))
            bias4 = tf.get_variable("b8-4",initializer=tf.constant(reader.get_tensor(('fusion_model/layer8/b8-4'))))
            CONV8M2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV8M1, weights4, strides=[1, 1, 1, 1], padding='SAME') + bias4, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            # RIGHT
            # 1
            weights5 = tf.get_variable("w8_5", initializer=tf.constant(reader.get_tensor('fusion_model/layer8/w8_5')))
            bias5 = tf.get_variable("b8-5",initializer=tf.constant(reader.get_tensor(('fusion_model/layer8/b8-5'))))
            CONV8R1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV8t, weights5, strides=[1, 1, 1, 1], padding='VALID') + bias5, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            CONV8R1 = lrelu(CONV8R1)
            # 2
            weights6 = tf.get_variable("w8_6", initializer=tf.constant(reader.get_tensor('fusion_model/layer8/w8_6')))
            bias6 = tf.get_variable("b8-6", initializer=tf.constant(reader.get_tensor(('fusion_model/layer8/b8-6'))))
            CONV8R2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV8R1, weights6, strides=[1, 1, 1, 1], padding='SAME') + bias6, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            CONV8R2 = lrelu(CONV8R2)
            # 3
            weights7 = tf.get_variable("w8_7", initializer=tf.constant(reader.get_tensor('fusion_model/layer8/w8_7')))
            bias7 = tf.get_variable("b8-7",initializer=tf.constant(reader.get_tensor(('fusion_model/layer8/b8-7'))))
            CONV8R3 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV8R2, weights7, strides=[1, 1, 1, 1], padding='SAME') + bias7, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            OUT8 = tf.concat([CONV8L, CONV8M2, CONV8R3], axis=-1)
            # bottom
            weights8 = tf.get_variable("w8_8", initializer=tf.constant(reader.get_tensor('fusion_model/layer8/w8_8')))
            bias8 = tf.get_variable("b8-8", initializer=tf.constant(reader.get_tensor(('fusion_model/layer8/b8-8'))))
            convb = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(OUT8, weights8, strides=[1, 1, 1, 1], padding='VALID') + bias8, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            convb =convb +CONV8t
            convb8 = lrelu(convb)
        with tf.variable_scope('layer9'):
            # TOP
            weights1 = tf.get_variable("w9_1",initializer=tf.constant(reader.get_tensor('fusion_model/layer9/w9_1')))
            bias1 = tf.get_variable("b9-1", initializer=tf.constant(reader.get_tensor(('fusion_model/layer9/b9-1'))))
            CONV9t = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(convb8, weights1, strides=[1, 1, 1, 1], padding='SAME') + bias1, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            CONV9t = lrelu(CONV9t)
            # LEFT
            weights2 = tf.get_variable("w9_2", initializer=tf.constant(reader.get_tensor('fusion_model/layer9/w9_2')))
            bias2 = tf.get_variable("b9-2", initializer=tf.constant(reader.get_tensor(('fusion_model/layer9/b9-2'))))
            CONV9L = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV9t, weights2, strides=[1, 1, 1, 1], padding='VALID') + bias2, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            # MEDIUM
            # 1
            weights3 = tf.get_variable("w9_3", initializer=tf.constant(reader.get_tensor('fusion_model/layer9/w9_3')))
            bias3 = tf.get_variable("b9-3", initializer=tf.constant(reader.get_tensor(('fusion_model/layer9/b9-3'))))
            CONV9M1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV9t, weights3, strides=[1, 1, 1, 1], padding='VALID') + bias3, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            CONV9M1 = lrelu(CONV9M1)
            # 2
            weights4 = tf.get_variable("w9_4", initializer=tf.constant(reader.get_tensor('fusion_model/layer9/w9_4')))
            bias4 = tf.get_variable("b9-4",  initializer=tf.constant(reader.get_tensor(('fusion_model/layer9/b9-4'))))
            CONV9M2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV9M1, weights4, strides=[1, 1, 1, 1], padding='SAME') + bias4, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            # RIGHT
            # 1
            weights5 = tf.get_variable("w9_5",initializer=tf.constant(reader.get_tensor('fusion_model/layer9/w9_5')))
            bias5 = tf.get_variable("b9-5", initializer=tf.constant(reader.get_tensor(('fusion_model/layer9/b9-5'))))
            CONV9R1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV9t, weights5, strides=[1, 1, 1, 1], padding='VALID') + bias5, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            CONV9R1 = lrelu(CONV9R1)
            # 2
            weights6 = tf.get_variable("w9_6",initializer=tf.constant(reader.get_tensor('fusion_model/layer9/w9_6')))
            bias6 = tf.get_variable("b9-6",  initializer=tf.constant(reader.get_tensor(('fusion_model/layer9/b9-6'))))
            CONV9R2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV9R1, weights6, strides=[1, 1, 1, 1], padding='SAME') + bias6, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            CONV9R2 = lrelu(CONV9R2)
            # 3
            weights7 = tf.get_variable("w9_7", initializer=tf.constant(reader.get_tensor('fusion_model/layer9/w9_7')))
            bias7 = tf.get_variable("b9-7", initializer=tf.constant(reader.get_tensor(('fusion_model/layer9/b9-7'))))
            CONV9R3 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV9R2, weights7, strides=[1, 1, 1, 1], padding='SAME') + bias7, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            OUT9 = tf.concat([CONV9L, CONV9M2, CONV9R3], axis=-1)
            # bottom
            weights8 = tf.get_variable("w9_8", initializer=tf.constant(reader.get_tensor('fusion_model/layer9/w9_8')))
            bias8 = tf.get_variable("b9-8", initializer=tf.constant(reader.get_tensor(('fusion_model/layer9/b9-8'))))
            convb6 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(OUT9, weights8, strides=[1, 1, 1, 1], padding='VALID') + bias8, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            convb6 =convb6 +CONV9t
            convb9 = lrelu(convb6)
        l9 = l7+ convb9+in4
        with tf.variable_scope('layer10'):
            # TOP
            weights1 = tf.get_variable("w10_1",  initializer=tf.constant(reader.get_tensor('fusion_model/layer10/w10_1')))
            bias1 = tf.get_variable("b10-1", initializer=tf.constant(reader.get_tensor(('fusion_model/layer10/b10-1'))))
            CONV10t = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(l9, weights1, strides=[1, 1, 1, 1], padding='SAME') + bias1, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            CONV10t = lrelu(CONV10t)
            # LEFT
            weights2 = tf.get_variable("w10_2",  initializer=tf.constant(reader.get_tensor('fusion_model/layer10/w10_2')))
            bias2 = tf.get_variable("b10-2", initializer=tf.constant(reader.get_tensor(('fusion_model/layer10/b10-2'))))
            CONV10L = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV10t, weights2, strides=[1, 1, 1, 1], padding='VALID') + bias2, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            # MEDIUM
            # 1
            weights3 = tf.get_variable("w10_3", initializer=tf.constant(reader.get_tensor('fusion_model/layer10/w10_3')))
            bias3 = tf.get_variable("b10-3",initializer=tf.constant(reader.get_tensor(('fusion_model/layer10/b10-3'))))
            CONV10M1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV10t, weights3, strides=[1, 1, 1, 1], padding='VALID') + bias3, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            CONV10M1 = lrelu(CONV10M1)
            # 2
            weights4 = tf.get_variable("w10_4", initializer=tf.constant(reader.get_tensor('fusion_model/layer10/w10_4')))
            bias4 = tf.get_variable("b10-4",initializer=tf.constant(reader.get_tensor(('fusion_model/layer10/b10-4'))))
            CONV10M2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV10M1, weights4, strides=[1, 1, 1, 1], padding='SAME') + bias4, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            # RIGHT
            # 1
            weights5 = tf.get_variable("w10_5", initializer=tf.constant(reader.get_tensor('fusion_model/layer10/w10_5')))
            bias5 = tf.get_variable("b10-5",initializer=tf.constant(reader.get_tensor(('fusion_model/layer10/b10-5'))))
            CONV10R1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV10t, weights5, strides=[1, 1, 1, 1], padding='VALID') + bias5, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            CONV10R1 = lrelu(CONV10R1)
            # 2
            weights6 = tf.get_variable("w10_6", initializer=tf.constant(reader.get_tensor('fusion_model/layer10/w10_6')))
            bias6 = tf.get_variable("b10-6", initializer=tf.constant(reader.get_tensor(('fusion_model/layer10/b10-6'))))
            CONV10R2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV10R1, weights6, strides=[1, 1, 1, 1], padding='SAME') + bias6, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            CONV10R2 = lrelu(CONV10R2)
            # 3
            weights7 = tf.get_variable("w10_7", initializer=tf.constant(reader.get_tensor('fusion_model/layer10/w10_7')))
            bias7 = tf.get_variable("b10-7",initializer=tf.constant(reader.get_tensor(('fusion_model/layer10/b10-7'))))
            CONV10R3 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV10R2, weights7, strides=[1, 1, 1, 1], padding='SAME') + bias7, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            OUT10 = tf.concat([CONV10L, CONV10M2, CONV10R3], axis=-1)
            # bottom
            weights8 = tf.get_variable("w10_8", initializer=tf.constant(reader.get_tensor('fusion_model/layer10/w10_8')))
            bias8 = tf.get_variable("b10-8", initializer=tf.constant(reader.get_tensor(('fusion_model/layer10/b10-8'))))
            convb = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(OUT10, weights8, strides=[1, 1, 1, 1], padding='VALID') + bias8, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            convb =convb +CONV10t
            convb10 = lrelu(convb)
        with tf.variable_scope('layer11'):
            # TOP
            weights1 = tf.get_variable("w11_1",initializer=tf.constant(reader.get_tensor('fusion_model/layer11/w11_1')))
            bias1 = tf.get_variable("b11-1", initializer=tf.constant(reader.get_tensor(('fusion_model/layer11/b11-1'))))
            CONV11t = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(convb10, weights1, strides=[1, 1, 1, 1], padding='SAME') + bias1, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            CONV11t = lrelu(CONV11t)
            # LEFT
            weights2 = tf.get_variable("w11_2", initializer=tf.constant(reader.get_tensor('fusion_model/layer11/w11_2')))
            bias2 = tf.get_variable("b11-2", initializer=tf.constant(reader.get_tensor(('fusion_model/layer11/b11-2'))))
            CONV11L = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV11t, weights2, strides=[1, 1, 1, 1], padding='VALID') + bias2, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            # MEDIUM
            # 1
            weights3 = tf.get_variable("w11_3", initializer=tf.constant(reader.get_tensor('fusion_model/layer11/w11_3')))
            bias3 = tf.get_variable("b11-3", initializer=tf.constant(reader.get_tensor(('fusion_model/layer11/b11-3'))))
            CONV11M1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV11t, weights3, strides=[1, 1, 1, 1], padding='VALID') + bias3, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            CONV11M1 = lrelu(CONV11M1)
            # 2
            weights4 = tf.get_variable("w11_4", initializer=tf.constant(reader.get_tensor('fusion_model/layer11/w11_4')))
            bias4 = tf.get_variable("b11-4",  initializer=tf.constant(reader.get_tensor(('fusion_model/layer11/b11-4'))))
            CONV11M2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV11M1, weights4, strides=[1, 1, 1, 1], padding='SAME') + bias4, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            # RIGHT
            # 1
            weights5 = tf.get_variable("w11_5",initializer=tf.constant(reader.get_tensor('fusion_model/layer11/w11_5')))
            bias5 = tf.get_variable("b11-5", initializer=tf.constant(reader.get_tensor(('fusion_model/layer11/b11-5'))))
            CONV11R1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV11t, weights5, strides=[1, 1, 1, 1], padding='VALID') + bias5, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            CONV11R1 = lrelu(CONV11R1)
            # 2
            weights6 = tf.get_variable("w11_6",initializer=tf.constant(reader.get_tensor('fusion_model/layer11/w11_6')))
            bias6 = tf.get_variable("b11-6",  initializer=tf.constant(reader.get_tensor(('fusion_model/layer11/b11-6'))))
            CONV11R2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV11R1, weights6, strides=[1, 1, 1, 1], padding='SAME') + bias6, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            CONV11R2 = lrelu(CONV11R2)
            # 3
            weights7 = tf.get_variable("w11_7", initializer=tf.constant(reader.get_tensor('fusion_model/layer11/w11_7')))
            bias7 = tf.get_variable("b11-7", initializer=tf.constant(reader.get_tensor(('fusion_model/layer11/b11-7'))))
            CONV11R3 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV11R2, weights7, strides=[1, 1, 1, 1], padding='SAME') + bias7, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            OUT11 = tf.concat([CONV11L, CONV11M2, CONV11R3], axis=-1)
            # bottom
            weights8 = tf.get_variable("w11_8", initializer=tf.constant(reader.get_tensor('fusion_model/layer11/w11_8')))
            bias8 = tf.get_variable("b11-8", initializer=tf.constant(reader.get_tensor(('fusion_model/layer11/b11-8'))))
            convb11 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(OUT11, weights8, strides=[1, 1, 1, 1], padding='VALID') + bias8, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            convb11 =convb11 +CONV11t
            convb11 = lrelu(convb11)
        l11 = l9+ convb11
        with tf.variable_scope('layer12'):
            # TOP
            weights1 = tf.get_variable("w12_1",  initializer=tf.constant(reader.get_tensor('fusion_model/layer12/w12_1')))
            bias1 = tf.get_variable("b12-1", initializer=tf.constant(reader.get_tensor(('fusion_model/layer12/b12-1'))))
            CONV12t = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(l11, weights1, strides=[1, 1, 1, 1], padding='SAME') + bias1, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            CONV12t = lrelu(CONV12t)
            # LEFT
            weights2 = tf.get_variable("w12_2",  initializer=tf.constant(reader.get_tensor('fusion_model/layer12/w12_2')))
            bias2 = tf.get_variable("b12-2", initializer=tf.constant(reader.get_tensor(('fusion_model/layer12/b12-2'))))
            CONV12L = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV12t, weights2, strides=[1, 1, 1, 1], padding='VALID') + bias2, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            # MEDIUM
            # 1
            weights3 = tf.get_variable("w12_3", initializer=tf.constant(reader.get_tensor('fusion_model/layer12/w12_3')))
            bias3 = tf.get_variable("b12-3",initializer=tf.constant(reader.get_tensor(('fusion_model/layer12/b12-3'))))
            CONV12M1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV12t, weights3, strides=[1, 1, 1, 1], padding='VALID') + bias3, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            CONV12M1 = lrelu(CONV12M1)
            # 2
            weights4 = tf.get_variable("w12_4", initializer=tf.constant(reader.get_tensor('fusion_model/layer12/w12_4')))
            bias4 = tf.get_variable("b12-4",initializer=tf.constant(reader.get_tensor(('fusion_model/layer12/b12-4'))))
            CONV12M2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV12M1, weights4, strides=[1, 1, 1, 1], padding='SAME') + bias4, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            # RIGHT
            # 1
            weights5 = tf.get_variable("w12_5", initializer=tf.constant(reader.get_tensor('fusion_model/layer12/w12_5')))
            bias5 = tf.get_variable("b12-5",initializer=tf.constant(reader.get_tensor(('fusion_model/layer12/b12-5'))))
            CONV12R1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV12t, weights5, strides=[1, 1, 1, 1], padding='VALID') + bias5, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            CONV12R1 = lrelu(CONV12R1)
            # 2
            weights6 = tf.get_variable("w12_6", initializer=tf.constant(reader.get_tensor('fusion_model/layer12/w12_6')))
            bias6 = tf.get_variable("b12-6", initializer=tf.constant(reader.get_tensor(('fusion_model/layer12/b12-6'))))
            CONV12R2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV12R1, weights6, strides=[1, 1, 1, 1], padding='SAME') + bias6, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            CONV12R2 = lrelu(CONV12R2)
            # 3
            weights7 = tf.get_variable("w12_7", initializer=tf.constant(reader.get_tensor('fusion_model/layer12/w12_7')))
            bias7 = tf.get_variable("b12-7",initializer=tf.constant(reader.get_tensor(('fusion_model/layer12/b12-7'))))
            CONV12R3 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV12R2, weights7, strides=[1, 1, 1, 1], padding='SAME') + bias7, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            OUT12= tf.concat([CONV12L, CONV12M2, CONV12R3], axis=-1)
            # bottom
            weights8 = tf.get_variable("w12_8", initializer=tf.constant(reader.get_tensor('fusion_model/layer12/w12_8')))
            bias8 = tf.get_variable("b12-8", initializer=tf.constant(reader.get_tensor(('fusion_model/layer12/b12-8'))))
            convb = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(OUT12, weights8, strides=[1, 1, 1, 1], padding='VALID') + bias8, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            convb =convb +CONV12t
            convb12 = lrelu(convb)
        with tf.variable_scope('layer13'):
            # TOP
            weights1 = tf.get_variable("w13_1",initializer=tf.constant(reader.get_tensor('fusion_model/layer13/w13_1')))
            bias1 = tf.get_variable("b13-1", initializer=tf.constant(reader.get_tensor(('fusion_model/layer13/b13-1'))))
            CONV13t = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(convb12, weights1, strides=[1, 1, 1, 1], padding='SAME') + bias1, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            CONV13t = lrelu(CONV13t)
            # LEFT
            weights2 = tf.get_variable("w13_2", initializer=tf.constant(reader.get_tensor('fusion_model/layer13/w13_2')))
            bias2 = tf.get_variable("b13-2", initializer=tf.constant(reader.get_tensor(('fusion_model/layer13/b13-2'))))
            CONV13L = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV13t, weights2, strides=[1, 1, 1, 1], padding='VALID') + bias2, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            # MEDIUM
            # 1
            weights3 = tf.get_variable("w13_3", initializer=tf.constant(reader.get_tensor('fusion_model/layer13/w13_3')))
            bias3 = tf.get_variable("b13-3", initializer=tf.constant(reader.get_tensor(('fusion_model/layer13/b13-3'))))
            CONV13M1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV13t, weights3, strides=[1, 1, 1, 1], padding='VALID') + bias3, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            CONV13M1 = lrelu(CONV13M1)
            # 2
            weights4 = tf.get_variable("w13_4", initializer=tf.constant(reader.get_tensor('fusion_model/layer13/w13_4')))
            bias4 = tf.get_variable("b13-4",  initializer=tf.constant(reader.get_tensor(('fusion_model/layer13/b13-4'))))
            CONV13M2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV13M1, weights4, strides=[1, 1, 1, 1], padding='SAME') + bias4, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            # RIGHT
            # 1
            weights5 = tf.get_variable("w13_5",initializer=tf.constant(reader.get_tensor('fusion_model/layer13/w13_5')))
            bias5 = tf.get_variable("b13-5", initializer=tf.constant(reader.get_tensor(('fusion_model/layer13/b13-5'))))
            CONV13R1 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV13t, weights5, strides=[1, 1, 1, 1], padding='VALID') + bias5, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            CONV13R1 = lrelu(CONV13R1)
            # 2
            weights6 = tf.get_variable("w13_6",initializer=tf.constant(reader.get_tensor('fusion_model/layer13/w13_6')))
            bias6 = tf.get_variable("b13-6",  initializer=tf.constant(reader.get_tensor(('fusion_model/layer13/b13-6'))))
            CONV13R2 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV13R1, weights6, strides=[1, 1, 1, 1], padding='SAME') + bias6, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            CONV13R2 = lrelu(CONV13R2)
            # 3
            weights7 = tf.get_variable("w13_7", initializer=tf.constant(reader.get_tensor('fusion_model/layer13/w13_7')))
            bias7 = tf.get_variable("b13-7", initializer=tf.constant(reader.get_tensor(('fusion_model/layer13/b13-7'))))
            CONV13R3 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(CONV13R2, weights7, strides=[1, 1, 1, 1], padding='SAME') + bias7, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            OUT13 = tf.concat([CONV13L, CONV13M2, CONV13R3], axis=-1)
            # bottom
            weights8 = tf.get_variable("w13_8", initializer=tf.constant(reader.get_tensor('fusion_model/layer13/w13_8')))
            bias8 = tf.get_variable("b13-8", initializer=tf.constant(reader.get_tensor(('fusion_model/layer13/b13-8'))))
            convb13 = tf.contrib.layers.batch_norm(
                tf.nn.conv2d(OUT13, weights8, strides=[1, 1, 1, 1], padding='VALID') + bias8, decay=0.9,
                updates_collections=None, epsilon=1e-5, scale=True)
            convb13 =convb13 +CONV13t
            convb13= lrelu(convb13)
        l13 = l11+ convb13+l9+in4
        with tf.variable_scope('layer14'):
            weights = tf.get_variable("w14", initializer=tf.constant(reader.get_tensor('fusion_model/layer14/w14')))
            bias = tf.get_variable("b14",initializer=tf.constant(reader.get_tensor(('fusion_model/layer14/b14'))))
            convb14 = tf.nn.conv2d( l13 , weights, strides=[1, 1, 1, 1], padding='VALID') + bias
            convb14 = tf.nn.tanh(convb14)
    return convb14
def input_setup(index):
    sub_gfp_sequence = []
    sub_pci_sequence = []
    input_gfp = imread(data_gfp[index])
    input_ = rgb2yuv(input_gfp)
    input_gfpy = input_[:, :, 0]
    gfpu = input_[:, :, 1]
    gfpv = input_[:, :, 2]
    input_gfpy = (input_gfpy - 127.5)/127.5
    w,h=input_gfpy.shape
    input_gfpy=input_gfpy.reshape([w,h,1])
    input_pci = imread(data_pci[index])
    input_pciy = input_pci[:, :, 0]
    input_pciy = (input_pciy - 127.5) / 127.5
    w,h=input_pciy.shape
    input_pciy=input_pciy.reshape([w,h,1])
    sub_gfp_sequence.append(input_gfpy)
    sub_pci_sequence.append(input_pciy)
    train_data_gfp = np.asarray(sub_gfp_sequence)
    train_data_pci = np.asarray(sub_pci_sequence)
    return train_data_gfp, train_data_pci, gfpu, gfpv


reader = tf.train.NewCheckpointReader('./checkpoint/DPCN/DPCN.model-19')
with tf.name_scope('GFP_input'):
  images_gfp = tf.placeholder(tf.float32, [1,None,None,None], name='images_gfp')
with tf.name_scope('PCI_input'):
  images_pci = tf.placeholder(tf.float32, [1,None,None,None], name='images_pci')
with tf.name_scope('input'):
  input_image =tf.concat([images_gfp,images_pci],axis=-1)
with tf.name_scope('fusion'):
  fusion_image=fusion_model(images_gfp, images_pci)
with tf.Session() as sess:
  init_op=tf.global_variables_initializer()
  sess.run(init_op)
  data_gfp=prepare_data('Test_gfp')
  data_pci=prepare_data('Test_pci')
  for i in range(len(data_gfp)):
      train_data_gfp, train_data_pci, gfp_u, gfp_v = input_setup(i)
      result = sess.run(fusion_image, feed_dict={images_gfp: train_data_gfp, images_pci: train_data_pci})
      result = result * 127.5 + 127.5
      result = 255 * (result - np.min(result)) / (np.max(result) - np.min(result))
      result = result.squeeze()
      h, w = result.shape
      result_y = np.reshape(result, [h, w, 1])
      gfp_u = np.reshape(gfp_u, [h, w, 1])
      gfp_v = np.reshape(gfp_v, [h, w, 1])
      result_yuv = np.concatenate([result_y, gfp_u, gfp_v], axis=-1)
      result_rgb = yuv2rgb(result_yuv)
      image_path_result = os.path.join(os.getcwd(), 'result')
      imsave_path_rgb = os.path.join(os.getcwd(), 'result_rgb')
      if not os.path.exists(image_path_result):
          os.makedirs(image_path_result)
      if not os.path.exists(imsave_path_rgb):
          os.makedirs(imsave_path_rgb)
      kkk = int((i + 1) / 100)
      kk = int(((i + 1) - kkk * 100) / 10)
      k = i + 1 - kkk * 100 - kk * 10
      image_path_result_1 = os.path.join(image_path_result, str(kkk) + str(kk) + str(k) + ".bmp")
      imsave_path_rgb_1 = os.path.join(imsave_path_rgb, str(kkk) + str(kk) + str(k) + ".bmp")
      imsave(result_rgb, imsave_path_rgb_1)
print("Testing success")
tf.reset_default_graph()

