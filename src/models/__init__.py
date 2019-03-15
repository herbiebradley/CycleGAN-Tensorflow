from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

# TODO: Merge into dataset class
def get_batches_per_epoch(opt):
    trainA_size = len(os.listdir(os.path.join(opt.data_dir, 'trainA')))
    trainB_size = len(os.listdir(os.path.join(opt.data_dir, 'trainB')))
    batches_per_epoch = (trainA_size + trainB_size) // (2 * opt.batch_size) # floor(Avg dataset size / batch_size)
    return batches_per_epoch
