import tensorflow as tf
import numpy as np
import nn
import rules

class model:


  def __init__(self):

    self.train_rate = .1

    self.chess = tf.placeholder(shape=[rules.N, rules.N], dtype=np.float32, name='chess')

    self.tar_choise = tf.placeholder(shape=[rules.N, rules.N], dtype=np.float32, name='choise')

    with tf.variable_scope('model'):

      x = tf.pad(self.chess, [[0, 1], [0, 1]])
      x = tf.reshape(x, [1, rules.N + 1, rules.N + 1, 1])

      with tf.variable_scope('layer1'):
        x = nn.conv(x, [3, 3], 4, name='conv1')
        x = nn.conv(x, [3, 3], 8, name='conv2')
        x = nn.pool(x, [2, 2])
        x = nn.relu(x)

      with tf.variable_scope('layer2'):
        x = nn.conv(x, [3, 3], 16, name='conv1')
        x = nn.conv(x, [3, 3], 32, name='conv2')
        x = nn.pool(x, [2, 2])
        x = nn.relu(x)

      with tf.variable_scope('layer3'):
        x = nn.conv(x, [3, 3], 64, name='conv1')
        x = nn.conv(x, [3, 3], 128, name='conv2')
        x = nn.pool(x, [2, 2])
        x = nn.relu(x)

      with tf.variable_scope('layer4'):
        x = nn.conv(x, [3, 3], 256, name='conv1')
        x = nn.conv(x, [3, 3], 512, name='conv2')
        x = nn.pool(x, [2, 2])
        x = nn.relu(x)


      with tf.variable_scope('layer-4'):
        x = nn.unpool(x)
        x = nn.conv(x, [3, 3], 256, name='conv1')
        x = nn.conv(x, [3, 3], 128, name='conv2')
        x = nn.relu(x)

      with tf.variable_scope('layer-3'):
        x = nn.unpool(x)
        x = nn.conv(x, [3, 3], 64, name='conv1')
        x = nn.conv(x, [3, 3], 32, name='conv2')
        x = nn.relu(x)

      with tf.variable_scope('layer-2'):
        x = nn.unpool(x)
        x = nn.conv(x, [3, 3], 16, name='conv1')
        x = nn.conv(x, [3, 3], 8, name='conv2')
        x = nn.relu(x)

      with tf.variable_scope('layer-1'):
        x = nn.unpool(x)
        x = nn.conv(x, [3, 3], 4, name='conv1')
        x = nn.conv(x, [3, 3], 1, name='conv2')
        x = nn.relu(x)

      x = tf.reshape(x, [rules.N + 1, rules.N + 1])
      x = x[0:15,0:15]
      
    self.choise = x
      
    self.loss = tf.nn.l2_loss(self.choise - self.tar_choise)
      
    self.opt = tf.train.AdadeltaOptimizer(self.train_rate).minimize(self.loss)

    self.sess = tf.InteractiveSession()

    self.saver = tf.train.Saver(max_to_keep=25)

    self.initer = tf.global_variables_initializer()
