from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def discriminator_loss(disc_of_real_output, disc_of_gen_output, label_value=1, use_lsgan=True):
    if use_lsgan: # Use least squares loss
        real_loss = tf.reduce_mean(tf.squared_difference(disc_of_real_output, label_value))
        generated_loss = tf.reduce_mean(tf.squared_difference(disc_of_gen_output, 1-label_value))

        total_disc_loss = (real_loss + generated_loss) * 0.5 # * 0.5 slows down rate D learns compared to G

    else: # Use vanilla GAN loss
        real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(disc_of_real_output), logits=disc_of_real_output)
        generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(disc_of_gen_output), logits=disc_of_gen_output)

        total_disc_loss = real_loss + generated_loss

    return total_disc_loss

def generator_loss(disc_of_gen_output, label_value=1, use_lsgan=True):
    if use_lsgan: # Use least squares loss
        gen_loss = tf.reduce_mean(tf.squared_difference(disc_of_gen_output, label_value))

    else: # Use vanilla GAN loss
        gen_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(disc_generated_output), logits=disc_generated_output)
        #l1_loss = tf.reduce_mean(tf.abs(target - gen_output)) # Look up pix2pix loss

    return gen_loss

def cycle_consistency_loss(data, reconstructed, norm='l1'):
    if norm == 'l1':
        loss = tf.reduce_mean(tf.abs(reconstructed - data))
        return loss
    else:
        raise NotImplementedError #TODO: l2 norm option

def identity_loss(data, identity, norm='l1'):
    if norm == 'l1':
        loss = tf.reduce_mean(tf.abs(identity - data))
        return loss
    else:
        raise NotImplementedError #TODO: l2 norm option
