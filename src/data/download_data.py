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
    parser.add_argument("--dataset_id", type=str, default="horse2zebra",help="String identifying the dataset to download. Available datasets are: apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower, ae_photos")
    parser.add_argument('--data_dir', required=True, help='download data to this directory')
    opt = parser.parse_args()

    # TODO: Add code to check if dataset is already there.
    download_data(opt.dataset_id, opt.data_dir)
