from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf

from train import initialize_checkpoint, define_model, restore_from_checkpoint
from pipeline.data import load_test_data, save_images

project_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
checkpoint_dir = os.path.join(project_dir, 'saved_models', 'checkpoints')
dataset_id = 'facades'

def test(data, model, checkpoint_info, dataset_id):
    path_to_dataset = os.path.join(project_dir, 'data', 'raw', dataset_id + os.sep)
    generatedA = os.path.join(path_to_dataset, 'generatedA' + os.sep)
    generatedB = os.path.join(path_to_dataset, 'generatedB' + os.sep)
    genA2B = model['genA2B']
    genB2A = model['genB2A']

    checkpoint, checkpoint_dir = checkpoint_info
    restore_from_checkpoint(checkpoint, checkpoint_dir)
    test_datasetA, test_datasetB, testA_size, testB_size = data
    test_datasetA = iter(test_datasetA)
    test_datasetB = iter(test_datasetB)

    for imageB in range(testB_size):
        start = time.time()
        try:
            # Get next testing image:
            testB = test_datasetB.get_next()
        except tf.errors.OutOfRangeError:
            print("Error, run out of data")
            break
        genB2A_output = genB2A(testB)
        with tf.device("/cpu:0"):
            save_images(genB2A_output, save_dir=generatedA, image_index=imageB)
    print("Generating {} test A images finished in {} sec\n".format(testA_size, time.time()-start))

    for imageA in range(testA_size):
        start = time.time()
        try:
            # Get next testing image:
            testA = test_datasetA.get_next()
        except tf.errors.OutOfRangeError:
            print("Error, run out of data")
            break
        genA2B_output = genA2B(testA)
        with tf.device("/cpu:0"):
            save_images(genA2B_output, save_dir=generatedB, image_index=imageA)
    print("Generating {} test B images finished in {} sec\n".format(testB_size, time.time()-start))

if __name__ == "__main__":
    with tf.device("/cpu:0"): # Preprocess data on CPU for significant performance gains.
        data = load_test_data(dataset_id, project_dir)
    with tf.device("/gpu:0"):
        model = define_model(training=False)
        checkpoint_info = initialize_checkpoint(checkpoint_dir, model, training=False)
        test(data, model, checkpoint_info, dataset_id)
