from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

def get_batches_per_epoch(dataset_id, project_dir):
    path_to_dataset = os.path.join(project_dir, 'data', 'raw', dataset_id + os.sep)
    trainA_path = os.path.join(path_to_dataset, 'trainA')
    trainB_path = os.path.join(path_to_dataset, 'trainB')
    trainA_size = len(os.listdir(trainA_path))
    trainB_size = len(os.listdir(trainB_path))
    batches_per_epoch = (trainA_size + trainB_size) // (2 * batch_size) # floor(Average dataset size / batch_size)
    return batches_per_epoch
