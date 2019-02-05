from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

# TODO: Merge into dataset class.
def get_batches_per_epoch(dataset_id, project_dir, batch_size=1):
    path_to_dataset = os.path.join(project_dir, 'data', 'raw', dataset_id + os.sep)
    trainA_path = os.path.join(path_to_dataset, 'trainA')
    trainB_path = os.path.join(path_to_dataset, 'trainB')
    trainA_size = len(os.listdir(trainA_path))
    trainB_size = len(os.listdir(trainB_path))
    batches_per_epoch = (trainA_size + trainB_size) // (2 * batch_size) # floor(Avg dataset size / batch_size)
    return batches_per_epoch

# TODO: Merge into basemodel class?
def get_learning_rate(initial_learning_rate, global_step, batches_per_epoch, const_iterations=100, decay_iterations=100):
    global_step = global_step.numpy() / 3 # /3 because there are 3 gradient updates per batch.
    total_epochs = global_step // batches_per_epoch
    learning_rate_lambda = 1.0 - max(0, total_epochs - const_iterations) / float(decay_iterations + 1)
    return initial_learning_rate * max(0, learning_rate_lambda)
