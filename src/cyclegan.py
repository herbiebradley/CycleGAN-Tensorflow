from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import multiprocessing
import glob

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import utils
from models.losses import generator_loss, discriminator_loss, cycle_consistency_loss
from models.networks import Generator, Discriminator
from utils.image_history_buffer import ImageHistoryBuffer

tf.enable_eager_execution()

project_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
initial_learning_rate = 0.0002
batch_size = 1 # Set batch size to 4 or 16 if training multigpu
img_size = 256
cyc_lambda = 10
epochs = 35
trainA_path = os.path.join(project_dir, 'data', 'raw', 'horse2zebra', 'trainA')
trainB_path = os.path.join(project_dir, 'data', 'raw', 'horse2zebra', 'trainB')
trainA_size = len(os.listdir(trainA_path))
trainB_size = len(os.listdir(trainB_path))
batches_per_epoch = (trainA_size + trainB_size) // (2 * batch_size) # floor(Average dataset size / batch_size)

def load_images(image_file):
    image = tf.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_images(image, [img_size, img_size])
    image = (image / 127.5) - 1 # Transform image to [-1, 1]
    return image

def load_train_data(dataset_id, batch_size=batch_size):
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
    train_datasetA = train_datasetA.apply(tf.contrib.data.map_and_batch(lambda x: load_images(x),
                                                            batch_size=batch_size,
                                                            num_parallel_calls=threads,
                                                            drop_remainder=True))
    # Queue up a number of batches on CPU side
    train_datasetA = train_datasetA.prefetch(buffer_size=threads)
    # Queue up batches asynchronously onto the GPU.
    # As long as there is a pool of batches CPU side a GPU prefetch of 1 is fine.
    train_datasetA = train_datasetA.apply(tf.contrib.data.prefetch_to_device("/gpu:0", buffer_size=1))

    train_datasetB = tf.data.Dataset.list_files(trainB_path + os.sep + '*.jpg', shuffle=False)
    train_datasetB = train_datasetB.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=trainB_size))
    train_datasetB = train_datasetB.apply(tf.contrib.data.map_and_batch(lambda x: load_images(x),
                                                            batch_size=batch_size,
                                                            num_parallel_calls=threads,
                                                            drop_remainder=True))
    train_datasetB = train_datasetB.prefetch(buffer_size=threads)
    train_datasetB = train_datasetB.apply(tf.contrib.data.prefetch_to_device("/gpu:0", buffer_size=1))

    return train_datasetA, train_datasetB

def load_test_data(dataset_id):
    path_to_dataset = os.path.join(project_dir, 'data', 'raw', dataset_id + os.sep)
    testA_path = os.path.join(path_to_dataset, 'testA')
    testB_path = os.path.join(path_to_dataset, 'testB')
    testA_size = len(os.listdir(testA_path))
    testB_size = len(os.listdir(testB_path))
    threads = multiprocessing.cpu_count()

    test_datasetA = tf.data.Dataset.list_files(testA_path + os.sep + '*.jpg', shuffle=False)
    test_datasetA = test_datasetA.apply(tf.contrib.data.map_and_batch(lambda x: load_images(x),
                                                            batch_size=1,
                                                            num_parallel_calls=threads,
                                                            drop_remainder=False))
    test_datasetA = test_datasetA.prefetch(buffer_size=threads)
    test_datasetA = test_datasetA.apply(tf.contrib.data.prefetch_to_device("/gpu:0", buffer_size=1))

    test_datasetB = tf.data.Dataset.list_files(testB_path + os.sep + '*.jpg', shuffle=False)
    test_datasetB = test_datasetB.apply(tf.contrib.data.map_and_batch(lambda x: load_images(x),
                                                            batch_size=1,
                                                            num_parallel_calls=threads,
                                                            drop_remainder=False))
    test_datasetB = test_datasetB.prefetch(buffer_size=threads)
    test_datasetB = test_datasetB.apply(tf.contrib.data.prefetch_to_device("/gpu:0", buffer_size=1))

    return test_datasetA, test_datasetB, testA_size, testB_size

def save_images(image_to_save, save_dir, name_suffix):
    save_file = os.path.join(save_dir,'test' + str(name_suffix))
    image = tf.reshape(image_to_save, shape=[img_size, img_size, 3])
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)
    image_string = tf.image.encode_jpeg(image, quality=95, format='rgb')
    tf.write_file(save_file, image_string)

def define_checkpoint(checkpoint_dir, model, training=True):
    if not training:
        genA2B = model['genA2B']
        genB2A = model['genB2A']

        global_step = tf.train.get_or_create_global_step()
        checkpoint = tf.train.Checkpoint(genA2B=genA2B, genB2A=genB2A,
                                         global_step=global_step)
    else:
        nets, optimizers = model
        discA = nets['discA']
        discB = nets['discB']
        genA2B = nets['genA2B']
        genB2A = nets['genB2A']
        discA_opt = optimizers['discA_opt']
        discB_opt = optimizers['discB_opt']
        genA2B_opt = optimizers['genA2B_opt']
        genB2A_opt = optimizers['genB2A_opt']
        learning_rate = optimizers['learning_rate']

        global_step = tf.train.get_or_create_global_step()
        checkpoint = tf.train.Checkpoint(discA=discA, discB=discB, genA2B=genA2B,
                                         genB2A=genB2A, discA_opt=discA_opt,
                                         discB_opt=discB_opt, genA2B_opt=genA2B_opt,
                                         genB2A_opt=genB2A_opt, learning_rate=learning_rate,
                                         global_step=global_step)
    return checkpoint, checkpoint_dir

def restore_from_checkpoint(checkpoint, checkpoint_dir):
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint is not None:
        # Use assert_existing_objects_matched() instead of asset_consumed() here because
        # optimizers aren't initialized fully until first gradient update.
        # This will throw an exception if checkpoint does not restore the model weights.
        checkpoint.restore(latest_checkpoint).assert_existing_objects_matched()
        print("Checkpoint restored from ", latest_checkpoint)
        # Uncomment below to print full list of checkpoint metadata.
        #print(tf.contrib.checkpoint.object_metadata(latest_checkpoint))
    else:
        print("No checkpoint found, initializing model.")

def define_model(initial_learning_rate, training=True):
    if not training:
        genA2B = Generator(img_size=img_size)
        genB2A = Generator(img_size=img_size)
        return {'genA2B':genA2B, 'genB2A':genB2A}
    else:
        discA = Discriminator()
        discB = Discriminator()
        genA2B = Generator(img_size=img_size)
        genB2A = Generator(img_size=img_size)
        learning_rate = tf.contrib.eager.Variable(initial_learning_rate, dtype=tf.float32, name='learning_rate')
        discA_opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
        discB_opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
        genA2B_opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
        genB2A_opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5)

        nets = {'discA':discA, 'discB':discB, 'genA2B':genA2B, 'genB2A':genB2A}
        optimizers = {'discA_opt':discA_opt, 'discB_opt':discB_opt, 'genA2B_opt':genA2B_opt,
                      'genB2A_opt':genB2A_opt, 'learning_rate':learning_rate}
        return nets, optimizers

def test(data, model, checkpoint_info, dataset_id, create_dir=False):
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

    for test_step in range(testB_size):
        start = time.time()
        try:
            # Get next testing image:
            testB = next(test_datasetB)
        except tf.errors.OutOfRangeError:
            print("Error, run out of data")
            break
        genB2A_output = genB2A(testB, training=False)
        save_images(genB2A_output, generatedA, test_step)
    print("Generating test A images finished in {} sec\n".format(time.time()-start))

    for test_step in range(testA_size):
        start = time.time()
        try:
            # Get next testing image:
            testA = next(test_datasetA)
        except tf.errors.OutOfRangeError:
            print("Error, run out of data")
            break
        genA2B_output = genA2B(testA, training=False)
        save_images(genA2B_output, generatedB,test_step)
    print("Generating test B images finished in {} sec\n".format(time.time()-start))

def train(data, model, checkpoint_info, epochs, initial_learning_rate=initial_learning_rate):
    nets, optimizers = model
    discA = nets['discA']
    discB = nets['discB']
    genA2B = nets['genA2B']
    genB2A = nets['genB2A']
    discA_opt = optimizers['discA_opt']
    discB_opt = optimizers['discB_opt']
    genA2B_opt = optimizers['genA2B_opt']
    genB2A_opt = optimizers['genB2A_opt']
    learning_rate = optimizers['learning_rate']

    checkpoint, checkpoint_dir = checkpoint_info
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    restore_from_checkpoint(checkpoint, checkpoint_dir)
    log_dir = os.path.join(project_dir, 'saved_models', 'tensorboard')

    # Create a tf.data.Iterator from the Datasets:
    train_datasetA, train_datasetB = iter(data[0]), iter(data[1])
    discA_buffer = ImageHistoryBuffer(50, batch_size, img_size // 8) # / 8 for PatchGAN
    discB_buffer = ImageHistoryBuffer(50, batch_size, img_size // 8)
    global_step = tf.train.get_or_create_global_step()
    log_dir = os.path.join(project_dir, 'saved_models', 'tensorboard')
    #summary_writer = tf.contrib.summary.create_file_writer(log_dir)

    for epoch in range(epochs):
        #with summary_writer.as_default():
        start = time.time()
        for train_step in range(batches_per_epoch):
            # Record summaries every 100 train_steps; there are 4 gradient updates per step.
            #with tf.contrib.summary.record_summaries_every_n_global_steps(400, global_step=global_step):
            try:
                # Get next training batches:
                trainA = next(train_datasetA)
                trainB = next(train_datasetB)
            except tf.errors.OutOfRangeError:
                print("Error, run out of data")
                break
            with tf.GradientTape(persistent=True) as tape:
                # Gen output shape: (batch_size, img_size, img_size, 3)
                genA2B_output = genA2B(trainA, training=True)
                genB2A_output = genB2A(trainB, training=True)
                # Disc output shape: (batch_size, img_size/8, img_size/8, 1)
                discA_real = discA(trainA, training=True)
                discB_real = discB(trainB, training=True)

                discA_fake = discA(genB2A_output, training=True)
                discB_fake = discB(genA2B_output, training=True)
                # Sample from history buffer of 50 images:
                discA_fake_refined = discA_buffer.query(discA_fake)
                discB_fake_refined = discB_buffer.query(discB_fake)

                reconstructedA = genB2A(genA2B_output, training=True)
                reconstructedB = genA2B(genB2A_output, training=True)

                cyc_loss = cycle_consistency_loss(trainA, trainB, reconstructedA, reconstructedB)
                genA2B_loss = generator_loss(discB_fake_refined) + cyc_loss
                genB2A_loss = generator_loss(discA_fake_refined) + cyc_loss
                discA_loss = discriminator_loss(discA_real, discA_fake_refined)
                discB_loss = discriminator_loss(discB_real, discB_fake_refined)

            discA_gradients = tape.gradient(discA_loss, discA.variables)
            discB_gradients = tape.gradient(discB_loss, discB.variables)
            genA2B_gradients = tape.gradient(genA2B_loss, genA2B.variables)
            genB2A_gradients = tape.gradient(genB2A_loss, genB2A.variables)

            discA_opt.apply_gradients(zip(discA_gradients, discA.variables), global_step=global_step)
            discB_opt.apply_gradients(zip(discB_gradients, discB.variables), global_step=global_step)
            genA2B_opt.apply_gradients(zip(genA2B_gradients, genA2B.variables), global_step=global_step)
            genB2A_opt.apply_gradients(zip(genB2A_gradients, genB2A.variables), global_step=global_step)

            # Summaries
            #tf.contrib.summary.scalar('train_step', global_step // 4)
            #tf.contrib.summary.scalar('loss/genA2B', genA2B_loss)
            #tf.contrib.summary.scalar('loss/genB2A', genB2A_loss)
            #tf.contrib.summary.scalar('loss/discA', discA_loss)
            #tf.contrib.summary.scalar('loss/discB', discB_loss)
            #tf.contrib.summary.scalar('loss/cyc', cyc_loss)
            #tf.contrib.summary.scalar('learning_rate', learning_rate)

            #tf.contrib.summary.histogram('discA/real', discA_real)
            #tf.contrib.summary.histogram('discA/fake', discA_fake)
            #tf.contrib.summary.histogram('discB/real', discB_real)
            #tf.contrib.summary.histogram('discB/fake', discA_fake)

            #tf.contrib.summary.image('A/generated', genB2A_output)
            #tf.contrib.summary.image('A/generated', reconstructedA)
            #tf.contrib.summary.image('B/generated', genA2B_output)
            #tf.contrib.summary.image('B/generated', reconstructedB)

            if train_step % 100 == 0:
                # Here we do global step / 4 because there are 4 gradient updates per batch.
                print("Global Training Step: ", global_step.numpy() // 4)
                print("Epoch Training Step: ", train_step + 1)
        # Assign decayed learning rate:
        learning_rate.assign(utils.get_learning_rate(initial_learning_rate, global_step,
                                                     batches_per_epoch))
        print("Learning rate in epoch {} is: {}".format(global_step.numpy() // batches_per_epoch,
                                                        learning_rate.numpy()))
        # Checkpoint the model:
        if (epoch + 1) % 2 == 0:
            checkpoint_path = checkpoint.save(file_prefix=checkpoint_prefix)
            print("Checkpoint saved at ", checkpoint_path)
        print ("Time taken for epoch {} is {} sec\n".format(epoch + 1, time.time()-start))

if __name__ == "__main__":
    checkpoint_dir = os.path.join(project_dir, 'saved_models', 'checkpoints')
    with tf.device("/cpu:0"): # Preprocess data on CPU for significant performance gains.
        dataset_id = 'horse2zebra'
        data = load_test_data(dataset_id)
    with tf.device("/gpu:0"):
        model = define_model(initial_learning_rate=initial_learning_rate, training=False)
        checkpoint_info = define_checkpoint(checkpoint_dir, model, training=False)
        test(data, model, checkpoint_info, dataset_id)
