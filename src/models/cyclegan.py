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
        self.initial_learning_rate = initial_learning_rate
        self.cyc_lambda = cyc_lambda
        self.identity_lambda = identity_lambda

        self.genA2B = Generator(num_gen_filters, img_size=img_size)
        self.genB2A = Generator(num_gen_filters, img_size=img_size)

        if self.isTrain:
            self.discA = Discriminator(num_disc_filters)
            self.discB = Discriminator(num_disc_filters)
            self.learning_rate = tf.contrib.eager.Variable(initial_learning_rate,
                                            dtype=tf.float32, name='learning_rate')
            self.disc_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5)
            self.gen_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5)
            self.global_step = tf.train.get_or_create_global_step()
            # Initialize history buffers:
            self.discA_buffer = ImageHistoryBuffer(50, batch_size, img_size)
            self.discB_buffer = ImageHistoryBuffer(50, batch_size, img_size)
        # Restore latest checkpoint:
        self.initialize_checkpoint()
        self.restore_checkpoint()

    def initialize_checkpoint(self):
        if self.isTrain:
            self.checkpoint = tf.train.Checkpoint(discA=self.discA,
                                                  discB=self.discB,
                                                  genA2B=self.genA2B,
                                                  genB2A=self.genB2A,
                                                  disc_opt=self.disc_opt,
                                                  gen_opt=self.gen_opt,
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
        self.dataA = input_batch[0].get_next()
        self.dataB = input_batch[1].get_next()

    def forward(self):
        # Gen output shape: (batch_size, img_size, img_size, 3)
        self.fakeB = self.genA2B(self.dataA)
        self.reconstructedA = self.genB2A(self.fakeB)

        self.fakeA = self.genB2A(self.dataB)
        self.reconstructedB = self.genA2B(self.fakeA)

    def backward_D(self, netD, real, fake):
        # Disc output shape: (batch_size, img_size/8, img_size/8, 1)
        pred_real = netD(real)
        pred_fake = netD(tf.stop_gradient(fake)) # Detaches generator from D
        disc_loss = discriminator_loss(pred_real, pred_fake)
        return disc_loss

    def backward_discA(self):
        fake_A = self.discA_buffer.query(self.fakeA)
        discA_loss = self.backward_D(self.discA, self.dataA, fake_A)
        return discA_loss

    def backward_discB(self):
        fake_B = self.discB_buffer.query(self.fakeB)
        discB_loss = self.backward_D(self.discB, self.dataB, fake_B)
        return discB_loss

    def backward_G(self):
        if self.identity_lambda > 0:
            identityA = self.genB2A(self.dataA)
            id_lossA = identity_loss(self.dataA, identityA) * self.cyc_lambda * self.identity_lambda

            identityB = self.genA2B(self.dataB)
            id_lossB = identity_loss(self.dataB, identityB) * self.cyc_lambda * self.identity_lambda
        else:
            id_lossA, id_lossB = 0, 0

        genA2B_loss = generator_loss(self.discB(self.fakeB))
        genB2A_loss = generator_loss(self.discA(self.fakeA))

        cyc_lossA = cycle_consistency_loss(self.dataA, self.reconstructedA) * self.cyc_lambda
        cyc_lossB = cycle_consistency_loss(self.dataB, self.reconstructedB) * self.cyc_lambda

        gen_loss = genA2B_loss + genB2A_loss + cyc_lossA + cyc_lossB + id_lossA + id_lossB
        return gen_loss

    def optimize_parameters(self):
        for net in (self.discA, self.discB):
            for layer in net.layers:
                layer.trainable = False

        with tf.GradientTape() as genTape: # Upgrade to 1.12 for watching?
            genTape.watch([self.genA2B.variables, self.genB2A.variables])

            self.forward()
            gen_loss = self.backward_G()

        gen_variables = [self.genA2B.variables, self.genB2A.variables]
        gen_gradients = genTape.gradient(gen_loss, gen_variables)
        self.gen_opt.apply_gradients(list(zip(gen_gradients[0], gen_variables[0])) \
                                + list(zip(gen_gradients[1], gen_variables[1])),
                                global_step=self.global_step)

        for net in (self.discA, self.discB):
            for layer in net.layers:
                layer.trainable = True

        with tf.GradientTape(persistent=True) as discTape: # Try 2 disc tapes?
            discTape.watch([self.discA.variables, self.discB.variables])

            discA_loss = self.backward_discA()
            discB_loss = self.backward_discB()

        discA_gradients = discTape.gradient(discA_loss, self.discA.variables)
        discB_gradients = discTape.gradient(discB_loss, self.discB.variables)
        self.disc_opt.apply_gradients(zip(discA_gradients, self.discA.variables),
                                                    global_step=self.global_step)
        self.disc_opt.apply_gradients(zip(discB_gradients, self.discB.variables),
                                                    global_step=self.global_step)

    def save_model(self):
        checkpoint_prefix = os.path.join(self.checkpoint_dir, 'ckpt')
        checkpoint_path = self.checkpoint.save(file_prefix=checkpoint_prefix)
        print("Checkpoint saved at ", checkpoint_path)

    def update_learning_rate(self):
        raise NotImplementedError
