from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf

def download_data(download_location, dataset_id):
    path_to_zip = tf.keras.utils.get_file(datset_id + '.zip', cache_subdir=os.path.abspath(download_location),
        origin='https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/' + datset_id + '.zip',
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
    project_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    raw_data = os.path.join(project_dir, 'data', 'raw')
    download_data(download_location=raw_data, FLAGS.dataset_id)
    # Add code to check if dataset is already there.
