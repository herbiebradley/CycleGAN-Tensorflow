import tensorflow as tf
"""
This module defines all CycleGAN losses. Options are included for
LSGAN, WGAN, and RGAN.
"""

def discriminator_loss(disc_of_real_output, disc_of_gen_output, gan_mode='lsgan', label_value=1):
    if gan_mode == 'lsgan': # Use least squares loss
        real_loss = tf.reduce_mean(tf.squared_difference(disc_of_real_output, label_value))
        generated_loss = tf.reduce_mean(tf.squared_difference(disc_of_gen_output, 1-label_value))

        total_disc_loss = (real_loss + generated_loss) * 0.5 # * 0.5 slows down rate D learns compared to G

    elif gan_mode == 'wgangp': # WGAN-GP loss
        total_disc_loss = tf.reduce_mean(disc_of_gen_output) - tf.reduce_mean(disc_of_real_output)

    elif gan_mode == 'rgan': # RGAN with vanilla GAN loss
        real = disc_of_real_output - disc_of_gen_output
        fake = disc_of_gen_output - disc_of_real_output

        real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(real), logits=real)
        generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(fake), logits=fake)

        total_disc_loss = real_loss + generated_loss

    else: # Use vanilla GAN loss
        real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(disc_of_real_output), logits=disc_of_real_output)
        generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(disc_of_gen_output), logits=disc_of_gen_output)

        total_disc_loss = real_loss + generated_loss

    return total_disc_loss

def generator_loss(disc_of_real_output, disc_of_gen_output, gan_mode='lsgan', label_value=1):
    if gan_mode == 'lsgan': # Use least squares loss
        gen_loss = tf.reduce_mean(tf.squared_difference(disc_of_gen_output, label_value))

    elif gan_mode == 'wgangp': # WGAN-GP loss
        gen_loss = -tf.reduce_mean(disc_of_gen_output)

    elif gan_mode == 'rgan': # RGAN with vanilla GAN loss
        real = disc_of_real_output - disc_of_gen_output
        fake = disc_of_gen_output - disc_of_real_output

        real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(real), logits=real)
        generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(fake), logits=fake)
        gen_loss = real_loss + generated_loss

    else: # Use vanilla GAN loss
        gen_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(disc_of_gen_output), logits=disc_of_gen_output)

    return gen_loss

def cycle_loss(data, reconstructed):
    loss = tf.reduce_mean(tf.abs(reconstructed - data))
    return loss

def identity_loss(data, identity):
    loss = tf.reduce_mean(tf.abs(identity - data))
    return loss
