from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse

import tensorflow as tf

def download_data(dataset_id, download_location):
    path_to_zip = tf.keras.utils.get_file(dataset_id + '.zip', cache_subdir=download_location,
        origin='https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/' + dataset_id + '.zip',
        extract=True)
    os.remove(path_to_zip)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
      "--dataset_id",
      type=str,
      default="horse2zebra",
      help="String identifying the dataset to download. For example, \
      'horse2zebra', 'monet2photo', 'summer2winter_yosemite', 'apple2orange', etc")
    FLAGS, unparsed = parser.parse_known_args()
    project_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
    raw_data = os.path.join(project_dir, 'data', 'raw')
    download_data(FLAGS.dataset_id, download_location=raw_data)
    # TODO: Add code to check if dataset is already there.
