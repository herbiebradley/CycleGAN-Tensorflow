from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import multiprocessing

import tensorflow as tf

import models
from pipeline.data import load_train_data
from models.losses import generator_loss, discriminator_loss, cycle_consistency_loss, identity_loss
from models.networks import Generator, Discriminator
from utils.image_history_buffer import ImageHistoryBuffer

tf.enable_eager_execution()

"""Hyperparameters (TODO: Move to argparse)"""
project_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
dataset_id = 'facades'
initial_learning_rate = 0.0002
num_gen_filters = 64
num_disc_filters = 64
batch_size = 1 # Set batch size to 4 or 16 if training multigpu
img_size = 256
cyc_lambda = 10
identity_lambda = 0
if dataset_id == 'monet2photo':
    identity_lambda = 0.5
epochs = 50
save_epoch_freq = 5
batches_per_epoch = models.get_batches_per_epoch(dataset_id, project_dir)

def define_checkpoint(checkpoint_dir, model, training=True):
    if not training:
        genA2B = model['genA2B']
        genB2A = model['genB2A']
        global_step = tf.train.get_or_create_global_step()
        checkpoint = tf.train.Checkpoint(genA2B=genA2B, genB2A=genB2A, global_step=global_step)
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

def define_model(initial_learning_rate=0.0002, training=True):
    if not training:
        genA2B = Generator(num_gen_filters, img_size=img_size)
        genB2A = Generator(num_gen_filters, img_size=img_size)
        return {'genA2B':genA2B, 'genB2A':genB2A}
    else:
        discA = Discriminator(num_disc_filters)
        discB = Discriminator(num_disc_filters)
        genA2B = Generator(num_gen_filters, img_size=img_size)
        genB2A = Generator(num_gen_filters, img_size=img_size)
        learning_rate = tf.contrib.eager.Variable(initial_learning_rate, dtype=tf.float32, name='learning_rate')
        discA_opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
        discB_opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
        genA2B_opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
        genB2A_opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5)

        nets = {'discA':discA, 'discB':discB, 'genA2B':genA2B, 'genB2A':genB2A}
        optimizers = {'discA_opt':discA_opt, 'discB_opt':discB_opt, 'genA2B_opt':genA2B_opt,
                      'genB2A_opt':genB2A_opt, 'learning_rate':learning_rate}
        return nets, optimizers

def train(data, model, checkpoint_info, epochs):
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
    # Restore latest checkpoint:
    checkpoint, checkpoint_dir = checkpoint_info
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    restore_from_checkpoint(checkpoint, checkpoint_dir)
    # Create a tf.data.Iterator from the Datasets:
    train_datasetA, train_datasetB = iter(data[0]), iter(data[1])
    # Initialize history buffers:
    discA_buffer = ImageHistoryBuffer(50, batch_size, img_size // 8) # / 8 for PatchGAN
    discB_buffer = ImageHistoryBuffer(50, batch_size, img_size // 8)
    # Initialize or restore global step:
    global_step = tf.train.get_or_create_global_step()
    # Initialize Tensorboard summary writer:
    log_dir = os.path.join(project_dir, 'saved_models', 'tensorboard')
    summary_writer = tf.contrib.summary.create_file_writer(log_dir, flush_millis=10000)
    for epoch in range(epochs):
        with summary_writer.as_default():
            start = time.time()
            for train_step in range(batches_per_epoch):
                # Record summaries every 100 train_steps; there are 4 gradient updates per step.
                with tf.contrib.summary.record_summaries_every_n_global_steps(400, global_step=global_step):
                    try:
                        # Get next training batches:
                        trainA = train_datasetA.get_next()
                        trainB = train_datasetB.get_next()
                    except tf.errors.OutOfRangeError:
                        print("Error, run out of data")
                        break
                    with tf.GradientTape(persistent=True) as tape:
                        # Gen output shape: (batch_size, img_size, img_size, 3)
                        genA2B_output = genA2B(trainA)
                        genB2A_output = genB2A(trainB)
                        # Disc output shape: (batch_size, img_size/8, img_size/8, 1)
                        discA_real = discA(trainA)
                        discB_real = discB(trainB)

                        discA_fake = discA(genB2A_output)
                        discB_fake = discB(genA2B_output)
                        # Sample from history buffer of 50 images:
                        discA_fake_refined = discA_buffer.query(discA_fake)
                        discB_fake_refined = discB_buffer.query(discB_fake)

                        reconstructedA = genB2A(genA2B_output)
                        reconstructedB = genA2B(genB2A_output)
                        identityA, identityB = 0, 0
                        if dataset_id == 'monet2photo':
                            identityA = genB2A(trainA)
                            identityB = genA2B(trainB)

                        cyc_loss = cyc_lambda * cycle_consistency_loss(trainA, trainB, reconstructedA, reconstructedB)
                        id_loss = identity_lambda * cyc_lambda * identity_loss(trainA, trainB, identityA, identityB)
                        genA2B_loss = generator_loss(discB_fake_refined) + cyc_loss + id_loss
                        genB2A_loss = generator_loss(discA_fake_refined) + cyc_loss + id_loss
                        discA_loss = discriminator_loss(discA_real, discA_fake_refined)
                        discB_loss = discriminator_loss(discB_real, discB_fake_refined)
                        # Summaries for Tensorboard:
                        tf.contrib.summary.scalar('loss/genA2B', genA2B_loss)
                        tf.contrib.summary.scalar('loss/genB2A', genB2A_loss)
                        tf.contrib.summary.scalar('loss/discA', discA_loss)
                        tf.contrib.summary.scalar('loss/discB', discB_loss)
                        tf.contrib.summary.scalar('loss/cyc', cyc_loss)
                        tf.contrib.summary.scalar('loss/identity', id_loss)
                        tf.contrib.summary.scalar('learning_rate', learning_rate)
                        tf.contrib.summary.image('A/generated', genB2A_output)
                        tf.contrib.summary.image('A/reconstructed', reconstructedA)
                        tf.contrib.summary.image('B/generated', genA2B_output)
                        tf.contrib.summary.image('B/reconstructed', reconstructedB)

                    discA_gradients = tape.gradient(discA_loss, discA.variables)
                    discB_gradients = tape.gradient(discB_loss, discB.variables)
                    genA2B_gradients = tape.gradient(genA2B_loss, genA2B.variables)
                    genB2A_gradients = tape.gradient(genB2A_loss, genB2A.variables)
                    # Try chaining disc and gen parameters into 2 optimizers?
                    discA_opt.apply_gradients(zip(discA_gradients, discA.variables), global_step=global_step)
                    discB_opt.apply_gradients(zip(discB_gradients, discB.variables), global_step=global_step)
                    genA2B_opt.apply_gradients(zip(genA2B_gradients, genA2B.variables), global_step=global_step)
                    genB2A_opt.apply_gradients(zip(genB2A_gradients, genB2A.variables), global_step=global_step)
        # Assign decayed learning rate:
        learning_rate.assign(models.get_learning_rate(initial_learning_rate, global_step, batches_per_epoch))
        # Checkpoint the model:
        if (epoch + 1) % save_epoch_freq == 0:
            checkpoint_path = checkpoint.save(file_prefix=checkpoint_prefix)
            print("Checkpoint saved at ", checkpoint_path)
        print("Global Training Step: ", global_step.numpy() // 4)
        print ("Time taken for local epoch {} is {} sec\n".format(global_step.numpy() // (4 * batches_per_epoch),
                                                                  time.time()-start))

if __name__ == "__main__":
    checkpoint_dir = os.path.join(project_dir, 'saved_models', 'checkpoints')
    with tf.device("/cpu:0"): # Preprocess data on CPU for significant performance gains.
        data = load_train_data(dataset_id, project_dir)
    #with tf.device("/gpu:0"):
        model = define_model(initial_learning_rate, training=True)
        checkpoint_info = define_checkpoint(checkpoint_dir, model, training=True)
        train(data, model, checkpoint_info, epochs=epochs)
