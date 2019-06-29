import os
import time

import tensorflow as tf

from utils.options import Options
from data.dataset import Dataset
from models.cyclegan import CycleGANModel

tf.enable_eager_execution()
"""
Run this module for testing.
Required args: --data_dir, --save_dir
"""
if __name__ == "__main__":
    opt = Options().parse(training=False)
    dataset = Dataset(opt)
    model = CycleGANModel(opt)

    device = ("/gpu:" + str(opt.gpu_id)) if opt.gpu_id != -1 else "/cpu:0"

    with tf.device(device):
        start = time.time()
        for image_index in range(opt.num_test):
            model.set_input(dataset.data)
            test_images = model.test()
            dataset.save_images(test_images, image_index)
        print("Generating {} test images for both datasets finished in {} sec\n".format(opt.num_test, time.time()-start))
