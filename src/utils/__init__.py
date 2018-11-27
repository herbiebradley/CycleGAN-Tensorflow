from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def get_learning_rate(initial_learning_rate, global_step, batches_per_epoch, const_iterations=100, decay_iterations=100):
    global_step = global_step.numpy() / 4 # Since there are 4 gradient updates per batch.
    total_epochs = global_step // batches_per_epoch
    learning_rate_lambda = 1.0 - max(0, total_epochs - const_iterations) / float(decay_iterations + 1)
    return initial_learning_rate * max(0, learning_rate_lambda)
