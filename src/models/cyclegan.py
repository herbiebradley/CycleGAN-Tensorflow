import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.enable_eager_execution()

import os
import time
import multiprocessing
import glob

""" Define Hyperparameters"""
project_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir + os.sep + os.pardir))
learning_rate = 0.0002
batch_size = 1 # Set batch size to 4 or 16 if training multigpu
img_size = 256
cyc_lambda = 10
epochs = 5
trainA_path = os.path.join(project_dir, "data", "raw", "horse2zebra", "trainA")
trainB_path = os.path.join(project_dir, "data", "raw", "horse2zebra", "trainB")
trainA_size = len(os.listdir(trainA_path))
trainB_size = len(os.listdir(trainB_path))
batches_per_epoch = int((trainA_size + trainB_size) / (2* batch_size)) # Average dataset size / batch_size

""" Load Datasets"""

def load_image(image_file):
    image = tf.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_images(image, [img_size, img_size])
    image = (image / 127.5) - 1 #Transform image to [-1, 1]
    return image

def download_data(download_location):
    path_to_zip = tf.keras.utils.get_file("horse2zebra.zip", cache_subdir=os.path.abspath(download_location),
        origin="https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip",
        extract=True)
    os.remove(path_to_zip)

def load_data(batch_size=batch_size, download=False):
    raw_data = os.path.join(project_dir, "data", "raw")
    if download:
        download_data(download_location=raw_data)

    path_to_dataset = os.path.join(raw_data, "horse2zebra/")
    trainA_path = os.path.join(path_to_dataset, "trainA")
    trainB_path = os.path.join(path_to_dataset, "trainB")

    trainA_size = len(os.listdir(trainA_path))
    trainB_size = len(os.listdir(trainB_path))
    threads = multiprocessing.cpu_count()

    train_datasetA = tf.data.Dataset.list_files(trainA_path + os.sep + "*.jpg", shuffle=False)
    train_datasetA = train_datasetA.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=trainA_size))
    train_datasetA = train_datasetA.apply(tf.contrib.data.map_and_batch(lambda x: load_image(x),
                                                            batch_size=batch_size,
                                                            num_parallel_calls=threads))
    train_datasetA = train_datasetA.prefetch(buffer_size=batch_size)

    train_datasetB = tf.data.Dataset.list_files(trainB_path + os.sep + "*.jpg", shuffle=False)
    train_datasetB = train_datasetB.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=trainB_size))
    train_datasetB = train_datasetB.apply(tf.contrib.data.map_and_batch(lambda x: load_image(x),
                                                            batch_size=batch_size,
                                                            num_parallel_calls=threads))
    train_datasetB = train_datasetB.prefetch(buffer_size=batch_size)

    return train_datasetA, train_datasetB

""" Define CycleGAN architecture"""

class Encoder(tf.keras.Model):

    def __init__(self):
        super(Encoder, self).__init__()

        # Small variance in initialization helps with preventing colour inversion.
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=7, strides=1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding="same", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.conv3 = tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding="same", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

    def call(self, inputs, training=True):
        x = tf.pad(inputs, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")

        x = self.conv1(x)
        x = tf.contrib.layers.instance_norm(x, epsilon=1e-05, trainable=training)
        # Implement instance norm to more closely match orig. paper (momentum=0.1)?
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = tf.contrib.layers.instance_norm(x, epsilon=1e-05, trainable=training)
        x = tf.nn.relu(x)

        x = self.conv3(x)
        x = tf.contrib.layers.instance_norm(x, epsilon=1e-05, trainable=training)
        x = tf.nn.relu(x)
        return x


class Residual(tf.keras.Model):

    def __init__(self):
        super(Residual, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(128, kernel_size=3, strides=1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.conv2 = tf.keras.layers.Conv2D(128, kernel_size=3, strides=1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

    def call(self, inputs, training=True):
        x = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")

        x = self.conv1(x)
        x = tf.contrib.layers.instance_norm(x, epsilon=1e-05, trainable=training)
        x = tf.nn.relu(x)

        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")

        x = self.conv2(x)
        x = tf.contrib.layers.instance_norm(x, epsilon=1e-05, trainable=training)

        x = tf.add(x, inputs)
        return x


class Decoder(tf.keras.Model):

    def __init__(self):
        super(Decoder, self).__init__()

        self.conv1 = tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding="same", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.conv2 = tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding="same", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.conv3 = tf.keras.layers.Conv2D(3, kernel_size=7, strides=1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

    def call(self, inputs, training=True):
        x = self.conv1(inputs)
        x = tf.contrib.layers.instance_norm(x, epsilon=1e-05, trainable=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = tf.contrib.layers.instance_norm(x, epsilon=1e-05, trainable=training)
        x = tf.nn.relu(x)

        x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")

        x = self.conv3(x)
        x = tf.contrib.layers.instance_norm(x, epsilon=1e-05, trainable=training)
        x = tf.nn.tanh(x)
        return x


class Generator(tf.keras.Model):

    def __init__(self, img_size=256, skip=False):
        super(Generator, self).__init__()

        self.img_size = img_size
        self.skip = skip #TODO: Add skip

        self.encoder = Encoder()
        if(img_size == 128):
            self.res1 = Residual()
            self.res2 = Residual()
            self.res3 = Residual()
            self.res4 = Residual()
            self.res5 = Residual()
            self.res6 = Residual()
        else:
            self.res1 = Residual()
            self.res2 = Residual()
            self.res3 = Residual()
            self.res4 = Residual()
            self.res5 = Residual()
            self.res6 = Residual()
            self.res7 = Residual()
            self.res8 = Residual()
            self.res9 = Residual()
        self.decoder = Decoder()

    @tf.contrib.eager.defun
    def call(self, inputs, training=True):
        x = self.encoder(inputs, training)
        if(img_size == 128):
            x = self.res1(x, training)
            x = self.res2(x, training)
            x = self.res3(x, training)
            x = self.res4(x, training)
            x = self.res5(x, training)
            x = self.res6(x, training)
        else:
            x = self.res1(x, training)
            x = self.res2(x, training)
            x = self.res3(x, training)
            x = self.res4(x, training)
            x = self.res5(x, training)
            x = self.res6(x, training)
            x = self.res7(x, training)
            x = self.res8(x, training)
            x = self.res9(x, training)
        x = self.decoder(x, training)
        return x

class Discriminator(tf.keras.Model):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, padding="same", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.conv2 = tf.keras.layers.Conv2D(128, kernel_size=4, strides=2, padding="same", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.conv3 = tf.keras.layers.Conv2D(256, kernel_size=4, strides=2, padding="same", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.conv4 = tf.keras.layers.Conv2D(512, kernel_size=4, strides=1, padding="same", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.conv5 = tf.keras.layers.Conv2D(1, kernel_size=4, strides=1, padding="same", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

        self.leaky = tf.keras.layers.LeakyReLU(0.2)

    @tf.contrib.eager.defun
    def call(self, inputs, training=True):
        x = self.conv1(inputs)
        x = self.leaky(x)

        x = self.conv2(x)
        x = tf.contrib.layers.instance_norm(x, epsilon=1e-05, trainable=training)
        x = self.leaky(x)

        x = self.conv3(x)
        x = tf.contrib.layers.instance_norm(x, epsilon=1e-05, trainable=training)
        x = self.leaky(x)

        x = self.conv4(x)
        x = tf.contrib.layers.instance_norm(x, epsilon=1e-05, trainable=training)
        x = self.leaky(x)

        x = self.conv5(x)
        #x = tf.nn.sigmoid(x) # use_sigmoid = not lsgan
        return x

"""Define Loss functions"""

def discriminator_loss(disc_of_real_output, disc_of_gen_output, lsgan=True):
    if lsgan: # Use least squares loss
        real_loss = tf.reduce_mean(tf.squared_difference(disc_of_real_output, 1))
        generated_loss = tf.reduce_mean(tf.square(disc_of_gen_output))

        total_disc_loss = (real_loss + generated_loss) * 0.5 # 0.5 slows down rate that D learns compared to G

    else: # Use vanilla GAN loss
        real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = tf.ones_like(disc_of_real_output), logits = disc_of_real_output)
        generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = tf.zeros_like(disc_of_gen_output), logits = disc_of_gen_output)

        total_disc_loss = real_loss + generated_loss

    return total_disc_loss

def generator_loss(disc_of_gen_output, lsgan=True):
    if lsgan: # Use least squares loss
        gen_loss = tf.reduce_mean(tf.squared_difference(disc_of_gen_output, 1))

    else: # Use vanilla GAN loss
        gen_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = tf.ones_like(disc_generated_output), logits = disc_generated_output)
        #l1_loss = tf.reduce_mean(tf.abs(target - gen_output)) # Look up pix2pix loss

    return gen_loss

def cycle_consistency_loss(dataA, dataB, reconstructed_dataA, reconstructed_dataB, cyc_lambda=10):
    loss = tf.reduce_mean(tf.abs(dataA - reconstructed_dataA) + tf.abs(dataB - reconstructed_dataB))
    return cyc_lambda * loss

def generate_images(fake_A, fake_B):
    plt.figure(figsize=(15,15))
    fake_A = tf.reshape(fake_A, [256, 256, 3])
    fake_B = tf.reshape(fake_B, [256, 256, 3])
    display_list = [fake_A, fake_B]
    title = ["Generated A", "Generated B"]
    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis("off")
    plt.show()

def define_checkpoint(checkpoint_dir, model):
    nets, optimizers = model
    discA = nets["discA"]
    discB = nets["discB"]
    genA2B = nets["genA2B"]
    genB2A = nets["genB2A"]
    discA_opt = optimizers["discA_opt"]
    discB_opt = optimizers["discB_opt"]
    genA2B_opt = optimizers["genA2B_opt"]
    genB2A_opt = optimizers["genB2A_opt"]

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(discA=discA, discB=discB, genA2B=genA2B, genB2A=genB2A,
                                 discA_opt=discA_opt, discB_opt=discB_opt,
                                 genA2B_opt=genA2B_opt, genB2A_opt=genB2A_opt,
                                 optimizer_step=tf.train.get_or_create_global_step())
    return checkpoint, checkpoint_dir, checkpoint_prefix

def restore_from_checkpoint(checkpoint, checkpoint_dir):
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint is not None:
        try:
            checkpoint.restore(latest_checkpoint).assert_consumed()
            print("Checkpoint restored")
        except:
            print("Restore failed")
    else:
        print("No checkpoint found, initializing model.")

def define_model(learning_rate, training=True): # Init only generators for testing
    discA = Discriminator()
    discB = Discriminator()
    genA2B = Generator()
    genB2A = Generator()
    discA_opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
    discB_opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
    genA2B_opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
    genB2A_opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5)

    nets = {"discA":discA, "discB":discB, "genA2B":genA2B, "genB2A":genB2A}
    optimizers = {"discA_opt":discA_opt, "discB_opt":discB_opt, "genA2B_opt":genA2B_opt, "genB2A_opt":genB2A_opt}
    return nets, optimizers

def test(data, model, checkpoints):
    nets, optimizers = model
    discA = nets["discA"]
    discB = nets["discB"]
    genA2B = nets["genA2B"]
    genB2A = nets["genB2A"]
    discA_opt = optimizers["discA_opt"]
    discB_opt = optimizers["discB_opt"]
    genA2B_opt = optimizers["genA2B_opt"]
    genB2A_opt = optimizers["genB2A_opt"]

    checkpoint, checkpoint_dir, checkpoint_prefix = checkpoints
    restore_from_checkpoint(checkpoint, checkpoint_dir)
    test_datasetA, test_datasetB = iter(data[0]), iter(data[1])

    for test_step in range(batches_per_epoch):
        start = time.time()
        try:
            # Get next testing minibatches
            testA = next(test_datasetA)
            testB = next(test_datasetB)
        except tf.errors.OutOfRangeError:
            print("Error, run out of data")
            break

        genA2B_output = genA2B(testA, training=False)
        genB2A_output = genB2A(testB, training=False)
        generate_images(genB2A_output, genA2genA2B_output)


def train(data, model, checkpoints, epochs, learning_rate=learning_rate, lsgan=True):
    nets, optimizers = model
    discA = nets["discA"]
    discB = nets["discB"]
    genA2B = nets["genA2B"]
    genB2A = nets["genB2A"]
    discA_opt = optimizers["discA_opt"]
    discB_opt = optimizers["discB_opt"]
    genA2B_opt = optimizers["genA2B_opt"]
    genB2A_opt = optimizers["genB2A_opt"]

    checkpoint, checkpoint_dir, checkpoint_prefix = checkpoints
    restore_from_checkpoint(checkpoint, checkpoint_dir)
    train_datasetA, train_datasetB = iter(data[0]), iter(data[1])

    for epoch in range(epochs):
        start = time.time()
        for train_step in range(batches_per_epoch):
            with tf.GradientTape() as genA2B_tape, tf.GradientTape() as genB2A_tape, \
                tf.GradientTape() as discA_tape, tf.GradientTape() as discB_tape:
                try:
                    # Get next training minibatches
                    trainA = next(train_datasetA)
                    trainB = next(train_datasetB)
                except tf.errors.OutOfRangeError:
                    print("Error, run out of data")
                    break

                genA2B_output = genA2B(trainA, training=True)
                genB2A_output = genB2A(trainB, training=True)

                discA_real_output = discA(trainA, training=True)
                discB_real_output = discB(trainB, training=True)

                discA_fake_output = discA(genB2A_output, training=True)
                discB_fake_output = discB(genA2B_output, training=True)

                reconstructedA = genB2A(genA2B_output, training=True)
                reconstructedB = genA2B(genB2A_output, training=True)

                discA_loss = discriminator_loss(discA_real_output, discA_fake_output, lsgan=lsgan)
                discB_loss = discriminator_loss(discB_real_output, discB_fake_output, lsgan=lsgan)
                genA2B_loss = generator_loss(discB_fake_output, lsgan=lsgan) + \
                              cycle_consistency_loss(trainA, trainB, reconstructedA, reconstructedB)
                genB2A_loss = generator_loss(discA_fake_output, lsgan=lsgan) + \
                              cycle_consistency_loss(trainA, trainB, reconstructedA, reconstructedB)

            genA2B_gradients = genA2B_tape.gradient(genA2B_loss, genA2B.variables)
            genB2A_gradients = genB2A_tape.gradient(genB2A_loss, genB2A.variables)
            discA_gradients = discA_tape.gradient(discA_loss, discA.variables)
            discB_gradients = discB_tape.gradient(discB_loss, discB.variables)

            genA2B_opt.apply_gradients(zip(genA2B_gradients, genA2B.variables),
                                       global_step=tf.train.get_or_create_global_step())
            genB2A_opt.apply_gradients(zip(genB2A_gradients, genB2A.variables),
                                       global_step=tf.train.get_or_create_global_step())
            discA_opt.apply_gradients(zip(discA_gradients, discA.variables),
                                      global_step=tf.train.get_or_create_global_step())
            discB_opt.apply_gradients(zip(discB_gradients, discB.variables),
                                      global_step=tf.train.get_or_create_global_step())

            if train_step % 100 == 0:
                print("Training step: ", train_step)
            # saving (checkpoint) the model
            #if (epoch + 1) % 3 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            print("Checkpoint saved at ", checkpoint_prefix)

        print ("Time taken for epoch {} is {} sec\n".format(epoch + 1, time.time()-start))

if __name__ == "__main__":
    checkpoint_dir = os.path.join(project_dir, "models", "checkpoints")
    with tf.device("/cpu:0"):
        data = load_data(batch_size=batch_size)
        model = define_model(learning_rate=learning_rate)
        checkpoints = define_checkpoint(checkpoint_dir, model)
    with tf.device("/gpu:0"):
        train(data, model, checkpoints, epochs=epochs, learning_rate=learning_rate, lsgan=True)
