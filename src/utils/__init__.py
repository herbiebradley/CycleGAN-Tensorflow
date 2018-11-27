from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import tensorflow as tf

def get_learning_rate(initial_learning_rate, global_step, batches_per_epoch, const_iterations=100, decay_iterations=100):
    global_step = global_step.numpy() / 4 # Since there are 4 gradient updates per batch.
    total_epochs = global_step // batches_per_epoch
    learning_rate_lambda = 1.0 - max(0, total_epochs - const_iterations) / float(decay_iterations + 1)
    return initial_learning_rate * max(0, learning_rate_lambda)

def generate_images(fake_A, fake_B):
    plt.figure(figsize=(15,15))
    fake_A = tf.reshape(fake_A, [256, 256, 3])
    fake_B = tf.reshape(fake_B, [256, 256, 3])
    display_list = [fake_A, fake_B]
    title = ["Generated A", "Generated B"]
    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()
