from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import multiprocessing

import tensorflow as tf
from PIL import Image

def load_image(image_file, img_size=256):
    # Read file into tensor of type string.
    image = tf.read_file(image_file)
    # Decodes file into jpg of type uint8 (range [0, 255]).
    image = tf.image.decode_jpeg(image, channels=3)
    # Convert to floating point with 32 bits (range [0, 1]).
    image = tf.image.convert_image_dtype(image, tf.float32)
    # TODO: Replace with PIL
    image = tf.image.resize_images(image, size=[img_size, img_size])
    image = tf.image.per_image_standardization(image)
    image = (image * 0.5) # Transform image to [-1, 1]
    return image

def save_images(image_to_save, save_dir, image_index):
    save_file = os.path.join(save_dir,'test' + str(image_index) + '.jpg')
    image = tf.reshape(image_to_save, shape=[img_size, img_size, 3])
    image = (image + 1) * 127.5 # Rescale images to [0, 255]
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)
    image_string = tf.image.encode_jpeg(image, format='rgb', quality=95)
    tf.write_file(save_file, image_string)

def load_train_data(dataset_id, project_dir, batch_size=1):
    path_to_dataset = os.path.join(project_dir, 'data', 'raw', dataset_id + os.sep)
    trainA_path = os.path.join(path_to_dataset, 'trainA')
    trainB_path = os.path.join(path_to_dataset, 'trainB')
    trainA_size = len(os.listdir(trainA_path))
    trainB_size = len(os.listdir(trainB_path))
    threads = multiprocessing.cpu_count()

    # Create Dataset from folder of string filenames.
    train_datasetA = tf.data.Dataset.list_files(trainA_path + os.sep + '*.jpg', shuffle=False)
    # Infinitely loop the dataset, shuffling once per epoch (in memory).
    # Safe to do since the dataset pipeline is currently string filenames.
    # Fused operation is faster than separated shuffle and repeat.
    # This is also serializable, so Dataset state can be saved with Checkpoints,
    # but doing this causes a segmentation fault for some reason...
    train_datasetA = train_datasetA.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=trainA_size))
    # Decodes filenames into jpegs, then stacks them into batches.
    # Throwing away the remainder allows the pipeline to report a fixed sized
    # batch size, aiding in model definition downstream.
    train_datasetA = train_datasetA.apply(tf.contrib.data.map_and_batch(lambda x: load_image(x),
                                                            batch_size=batch_size,
                                                            num_parallel_calls=threads,
                                                            drop_remainder=True))
    # Queue up a number of batches on CPU side
    train_datasetA = train_datasetA.prefetch(buffer_size=threads)
    # Queue up batches asynchronously onto the GPU.
    # As long as there is a pool of batches CPU side a GPU prefetch of 1 is fine.
    # TODO: If GPU exists:
    train_datasetA = train_datasetA.apply(tf.contrib.data.prefetch_to_device("/gpu:0", buffer_size=1))

    train_datasetB = tf.data.Dataset.list_files(trainB_path + os.sep + '*.jpg', shuffle=False)
    train_datasetB = train_datasetB.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=trainB_size))
    train_datasetB = train_datasetB.apply(tf.contrib.data.map_and_batch(lambda x: load_image(x),
                                                            batch_size=batch_size,
                                                            num_parallel_calls=threads,
                                                            drop_remainder=True))
    train_datasetB = train_datasetB.prefetch(buffer_size=threads)
    train_datasetB = train_datasetB.apply(tf.contrib.data.prefetch_to_device("/gpu:0", buffer_size=1))

    return train_datasetA, train_datasetB

def load_test_data(dataset_id, project_dir):
    path_to_dataset = os.path.join(project_dir, 'data', 'raw', dataset_id + os.sep)
    testA_path = os.path.join(path_to_dataset, 'testA')
    testB_path = os.path.join(path_to_dataset, 'testB')
    testA_size = len(os.listdir(testA_path))
    testB_size = len(os.listdir(testB_path))
    threads = multiprocessing.cpu_count()

    test_datasetA = tf.data.Dataset.list_files(testA_path + os.sep + '*.jpg', shuffle=False)
    test_datasetA = test_datasetA.apply(tf.contrib.data.map_and_batch(lambda x: load_image(x),
                                                            batch_size=1,
                                                            num_parallel_calls=threads,
                                                            drop_remainder=False))
    #test_datasetA = test_datasetA.prefetch(buffer_size=threads)
    #test_datasetA = test_datasetA.apply(tf.contrib.data.prefetch_to_device("/gpu:0", buffer_size=1))

    test_datasetB = tf.data.Dataset.list_files(testB_path + os.sep + '*.jpg', shuffle=False)
    test_datasetB = test_datasetB.apply(tf.contrib.data.map_and_batch(lambda x: load_image(x),
                                                            batch_size=1,
                                                            num_parallel_calls=threads,
                                                            drop_remainder=False))
    #test_datasetB = test_datasetB.prefetch(buffer_size=threads)
    #test_datasetB = test_datasetB.apply(tf.contrib.data.prefetch_to_device("/gpu:0", buffer_size=1))

    test_datasetA = iter(test_datasetB)
    testA = next(test_datasetA)
    print("A Max: ", tf.reduce_max(testA))
    print("A Min: ", tf.reduce_min(testA))
    print("A Mean: ", tf.reduce_mean(testA))
    print(testA)

    return test_datasetA, test_datasetB, testA_size, testB_size
