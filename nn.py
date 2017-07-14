import os
import sys

import numpy as np
import tensorflow as tf

LOSS = 0

variable_slist = []
variable_cnt = 0

def _get_variable(name, shape, wd=.0, stddev=0.01):
  global LOSS, variable_cnt, variable_slist

  with tf.device('cpu:0'):

    ret = tf.get_variable(name=name, shape=shape, initializer=tf.truncated_normal_initializer(
        dtype=tf.float32, mean=0.0, stddev=stddev))

    if wd != .0:
      LOSS += wd * tf.nn.l2_loss(ret)

    return ret

def relu(x):
  return tf.nn.elu(x)

def fc(x, o_shape, name='fc'):
  with tf.variable_scope(name):
    shape = x.get_shape().as_list()
    batch, shape = shape[0], shape[1:]

    size = 1
    for i in shape:
      size *= i

    o_size = 1
    for i in o_shape:
      o_size *= i

    k = _get_variable('weights', [size, o_size])
    b = _get_variable('biases', [1, o_size])

    x = tf.reshape(x, [batch, size])
    x = tf.matmul(x, k) + b
    x = tf.reshape(x, [batch] + o_shape)

    return x


def pool(x, window_shape=None, data_format=None, pooling_type='MAX', name='pool'):
  if not window_shape:
    ax_size = len(x.get_shape().as_list()) - 2
    window_shape = [2 for i in range(0, ax_size)]

  with tf.variable_scope(name):
    return tf.nn.pool(x, window_shape=window_shape, pooling_type=pooling_type, padding='SAME', strides=window_shape, data_format=data_format)


def unpool(x, name='unpool', shape=None):
  with tf.variable_scope(name):
    if not shape:
      shape = x.get_shape().as_list()
      shape = [2*shape[1], 2*shape[2]]
    else:
      shape = [shape[1], shape[2]]
    return tf.image.resize_nearest_neighbor(x, shape, name=name)


def conv(x, kernel, channel, data_format=None, name='conv'):
  with tf.variable_scope(name):

    shape = x.get_shape().as_list()

    if data_format == None or data_format[0:2] == 'NC':
      N, shape, C = shape[0], shape[1:-1], shape[-1]
      b = _get_variable('biases', [1] + [1 for i in range(0, len(shape))] + [channel])

    else:
      N, C, shape = shape[0], shape[1], shape[2:]
      b = _get_variable('biases', [1] + [channel] + [1 for i in range(0, len(shape))])

    k = _get_variable('weights', kernel + [C] + [channel])

    return tf.nn.convolution(x, k, padding='SAME', data_format=data_format) + b


def bn(x, axis=[], name='bn', eps=1e-8):
  '''
  tensorflow op: batch_normalization
  '''
  with tf.variable_scope(name):
    shape = x.get_shape().as_list()
    ax = []
    for i in range(0, len(shape)):
      if not i in axis:
        ax.append(i)
    mean, variance = tf.nn.moments(x, ax, keep_dims=True)
    return tf.nn.batch_normalization(x, mean, variance, None, None, eps)


def dropout(x, prob, name='dropout'):
  '''
  ERROR: not finished yet
  '''
  with tf.variable_scope(name):
    if TRAIN:
      return tf.nn.dropout(x, prob)
    else:
      return x


def layer_add(x, y, name='layer_add'):
  with tf.variable_scope(name):
    wx = _get_variable('weight_x', [1])
    wy = _get_variable('weight_y', [1])
    b = _get_variable('biases', [1])
    return wx * x + wy * y + b
