from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

class CycleGANModel(object):

    def __init__(self, initial_learning_rate, num_gen_filters, num_disc_filters, training=True):
        if not training:
            self.genA2B = Generator(num_gen_filters, img_size=img_size)
            self.genB2A = Generator(num_gen_filters, img_size=img_size)
        else:
            self.discA = Discriminator(num_disc_filters)
            self.discB = Discriminator(num_disc_filters)
            self.genA2B = Generator(num_gen_filters, img_size=img_size)
            self.genB2A = Generator(num_gen_filters, img_size=img_size)
            self.learning_rate = tf.contrib.eager.Variable(initial_learning_rate, dtype=tf.float32, name='learning_rate')
            self.discA_opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
            self.discB_opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
            self.genA2B_opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
            self.genB2A_opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5)

    def initialize_checkpoint(self):
        raise NotImplementedError

    def restore_checkpoint(self):
        raise NotImplementedError

    def train_one_epoch(self):
        raise NotImplementedError
