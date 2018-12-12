from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

import models
from models.losses import generator_loss, discriminator_loss, cycle_consistency_loss, identity_loss
from models.networks import Generator, Discriminator
from utils.image_history_buffer import ImageHistoryBuffer

class CycleGANModel(object):

    def __init__(self, initial_learning_rate, num_gen_filters, num_disc_filters,
                 batch_size, cyc_lambda, identity_lambda, checkpoint_dir,
                 img_size, training):
        self.isTrain = training
        self.checkpoint_dir = checkpoint_dir

        self.genA2B = Generator(num_gen_filters, img_size=img_size)
        self.genB2A = Generator(num_gen_filters, img_size=img_size)

        if self.isTrain:
            self.discA = Discriminator(num_disc_filters)
            self.discB = Discriminator(num_disc_filters)
            self.learning_rate = tf.contrib.eager.Variable(initial_learning_rate, dtype=tf.float32, name='learning_rate')
            self.discA_opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
            self.discB_opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
            self.genA2B_opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
            self.genB2A_opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
            self.global_step = tf.train.get_or_create_global_step()
            # Initialize history buffers:
            self.discA_buffer = ImageHistoryBuffer(50, batch_size, img_size // 8) # / 8 for PatchGAN
            self.discB_buffer = ImageHistoryBuffer(50, batch_size, img_size // 8)
        # Restore latest checkpoint:
        self.initialize_checkpoint()
        self.restore_checkpoint()


    def forward(self):
        raise NotImplementedError

    def optimize_parameters(self):
        raise NotImplementedError

    def initialize_checkpoint(self):
        if self.isTrain:
            self.checkpoint = tf.train.Checkpoint(discA=self.discA,
                                                  discB=self.discB,
                                                  genA2B=self.genA2B,
                                                  genB2A=self.genB2A,
                                                  discA_opt=self.discA_opt,
                                                  discB_opt=self.discB_opt,
                                                  genA2B_opt=self.genA2B_opt,
                                                  genB2A_opt=self.genB2A_opt,
                                                  learning_rate=self.learning_rate,
                                                  global_step=self.global_step)
        else:
            self.checkpoint = tf.train.Checkpoint(genA2B=self.genA2B,
                                                  genB2A=self.genB2A,
                                                  global_step=self.global_step)

    def restore_checkpoint(self):
        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest_checkpoint is not None:
            # Use assert_existing_objects_matched() instead of asset_consumed() here because
            # optimizers aren't initialized fully until first gradient update.
            # This will throw an exception if checkpoint does not restore the model weights.
            self.checkpoint.restore(latest_checkpoint).assert_existing_objects_matched()
            print("Checkpoint restored from ", latest_checkpoint)
            # Uncomment below to print full list of checkpoint metadata.
            #print(tf.contrib.checkpoint.object_metadata(latest_checkpoint))
        else:
            print("No checkpoint found, initializing model.")

    def load_batch(self, input_batch):
        self.realA = input_batch[0].get_next()
        self.realB = input_batch[1].get_next()

    def forward(self):
        self.fakeB = self.genA2B(self.realA)
        self.reconstructedA = self.genB2A(self.fakeB)

        self.fakeA = self.genB2A(self.realB)
        self.reconstructedB = self.genA2B(self.fakeA)

    def backward_D_basic(self, disc, real, fake):
        # Real
        pred_real = disc(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = disc(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def optimize_parameters(self):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError

    def update_learning_rate(self):
        raise NotImplementedError
