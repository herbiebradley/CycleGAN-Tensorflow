from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf

import models
from models.cyclegan import CycleGANModel
from pipeline.data import load_train_data

tf.enable_eager_execution()

"""Hyperparameters (TODO: Move to argparse)"""
project_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
dataset_id = 'facades'
epochs = 15
batches_per_epoch = models.get_batches_per_epoch(dataset_id, project_dir)
batch_size = 1 # Set batch size to 4 or 16 if training multigpu
initial_learning_rate = 0.0002
num_gen_filters = 32
num_disc_filters = 64
img_size = 256
cyc_lambda = 10
identity_lambda = 0.5
save_epoch_freq = 5
#tf.contrib.summary.scalar('loss/genA2B', genA2B_loss_basic)
#tf.contrib.summary.scalar('loss/genB2A', genB2A_loss_basic)
#tf.contrib.summary.scalar('loss/discA', discA_loss)
#tf.contrib.summary.scalar('loss/discB', discB_loss)
#tf.contrib.summary.scalar('loss/cyc', cyc_loss)
#tf.contrib.summary.scalar('loss/identity', id_loss)
#tf.contrib.summary.scalar('learning_rate', learning_rate)

def train_one_epoch():
    raise NotImplementedError

if __name__ == "__main__":
    checkpoint_dir = os.path.join(project_dir, 'saved_models', 'checkpoints')
    with tf.device("/cpu:0"):
        # Preprocess data on CPU for significant performance gains:
        dataA, dataB = load_train_data(dataset_id, project_dir)
        model = CycleGANModel(initial_learning_rate, num_gen_filters,
                                 num_disc_filters, batch_size, cyc_lambda,
                                 identity_lambda, checkpoint_dir, img_size,
                                 training=True)
    with tf.device("/gpu:0"):
        # Initialize Tensorboard summary writer:
        log_dir = os.path.join(project_dir, 'saved_models', 'tensorboard')
        summary_writer = tf.contrib.summary.create_file_writer(log_dir, flush_millis=10000)
        for epoch in range(epochs):
            start = time.time()
            with summary_writer.as_default():
                for train_step in range(batches_per_epoch):
                    # Record summaries every 100 train_steps; there are 3 gradient updates per step.
                    with tf.contrib.summary.record_summaries_every_n_global_steps(300, global_step=model.global_step):
                        # Get next training batches:
                        batch = {"A": dataA.get_next(), "B": dataB.get_next()}
                        model.set_input(batch)
                        model.optimize_parameters()
                        print("Iteration ", train_step)
                        # Summaries for Tensorboard:
                        tf.contrib.summary.image('A/generated', model.fakeA)
                        tf.contrib.summary.image('A/reconstructed', model.reconstructedA)
                        tf.contrib.summary.image('B/generated', model.fakeB)
                        tf.contrib.summary.image('B/reconstructed', model.reconstructedB)
            # Assign decayed learning rate:
            model.update_learning_rate(batches_per_epoch)
            # Checkpoint the model:
            if (epoch + 1) % save_epoch_freq == 0:
                model.save_model()
            print("Global Training Step: ", global_step.numpy() // 3)
            print ("Time taken for total epoch {} is {} sec\n".format(global_step.numpy() \
                                            // (3 * batches_per_epoch), time.time()-start))
