from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf

import models
from models.cyclegan import CycleGANModel
from data.dataset import load_train_data
from utils.options import Options

tf.enable_eager_execution()

def train_one_epoch(self):
    raise NotImplementedError

if __name__ == "__main__":
    opt = Options(isTrain=True)
    with tf.device("/cpu:0"):
        # Preprocess data on CPU for significant performance gains:
        dataA, dataB = load_train_data(dataset_id, project_dir)
        model = CycleGANModel(opt)
        batches_per_epoch = models.get_batches_per_epoch(opt)
    with tf.device("/gpu:0"):
        global_step = model.global_step
        # Initialize Tensorboard summary writer:
        log_dir = os.path.join(opt.save_dir, 'tensorboard')
        summary_writer = tf.contrib.summary.create_file_writer(log_dir, flush_millis=10000)
        for epoch in range(opt.epochs):
            start = time.time()
            with summary_writer.as_default():
                for train_step in range(batches_per_epoch):
                    # Record summaries every 100 train_steps, we multiply by 3 because there are 3 gradient updates per step.
                    with tf.contrib.summary.record_summaries_every_n_global_steps(opt.summary_freq * 3, global_step=global_step):
                        # Get next training batches:
                        batch = {"A": dataA.get_next(), "B": dataB.get_next()}
                        model.set_input(batch)
                        model.optimize_parameters()

                        # Summaries for Tensorboard:
                        tf.contrib.summary.image('A/generated', model.fakeA)
                        tf.contrib.summary.image('A/reconstructed', model.reconstructedA)
                        tf.contrib.summary.image('B/generated', model.fakeB)
                        tf.contrib.summary.image('B/reconstructed', model.reconstructedB)
                        #tf.contrib.summary.scalar('loss/genA2B', genA2B_loss_basic)
                        #tf.contrib.summary.scalar('loss/genB2A', genB2A_loss_basic)
                        #tf.contrib.summary.scalar('loss/discA', discA_loss)
                        #tf.contrib.summary.scalar('loss/discB', discB_loss)
                        #tf.contrib.summary.scalar('loss/cyc', cyc_loss)
                        #tf.contrib.summary.scalar('loss/identity', id_loss)
                        #tf.contrib.summary.scalar('learning_rate', learning_rate)
            # Assign decayed learning rate:
            model.update_learning_rate(batches_per_epoch)
            # Checkpoint the model:
            if (epoch + 1) % opt.save_epoch_freq == 0:
                model.save_model()
            print("Global Training Step: ", global_step.numpy() // 3)
            print ("Time taken for total epoch {} is {} sec\n".format(global_step.numpy() \
                                            // (3 * batches_per_epoch), time.time()-start))
