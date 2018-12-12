from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class Encoder(tf.keras.Model):

    def __init__(self, ngf):
        super(Encoder, self).__init__()
        # Small variance in initialization helps with preventing colour inversion.
        self.conv1 = tf.keras.layers.Conv2D(ngf, kernel_size=7, strides=1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.conv2 = tf.keras.layers.Conv2D(ngf * 2, kernel_size=3, strides=2, padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.conv3 = tf.keras.layers.Conv2D(ngf * 4, kernel_size=3, strides=2, padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

    def call(self, inputs):
        # Reflection padding is used to reduce artifacts.
        x = tf.pad(inputs, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT')
        x = self.conv1(x)
        x = tf.contrib.layers.instance_norm(x, center=False, scale=False, epsilon=1e-05, trainable=False)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = tf.contrib.layers.instance_norm(x, center=False, scale=False, epsilon=1e-05, trainable=False)
        x = tf.nn.relu(x)

        x = self.conv3(x)
        x = tf.contrib.layers.instance_norm(x, center=False, scale=False, epsilon=1e-05, trainable=False)
        x = tf.nn.relu(x)
        return x


class Residual(tf.keras.Model):

    def __init__(self, ngf):
        super(Residual, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(ngf * 4, kernel_size=3, strides=1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.conv2 = tf.keras.layers.Conv2D(ngf * 4, kernel_size=3, strides=1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

    def call(self, inputs):
        x = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        x = self.conv1(x)
        x = tf.contrib.layers.instance_norm(x, center=False, scale=False, epsilon=1e-05, trainable=False)
        x = tf.nn.relu(x)

        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        x = self.conv2(x)
        x = tf.contrib.layers.instance_norm(x, center=False, scale=False, epsilon=1e-05, trainable=False)

        x = tf.add(x, inputs) # Add is better than concatenation.
        return x


class Decoder(tf.keras.Model):

    def __init__(self, ngf):
        super(Decoder, self).__init__()

        self.conv1 = tf.keras.layers.Conv2DTranspose(ngf * 2, kernel_size=3, strides=2, padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.conv2 = tf.keras.layers.Conv2DTranspose(ngf, kernel_size=3, strides=2, padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.conv3 = tf.keras.layers.Conv2D(3, kernel_size=7, strides=1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

    def call(self, inputs):

        x = self.conv1(inputs)
        x = tf.contrib.layers.instance_norm(x, center=False, scale=False, epsilon=1e-05, trainable=False)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = tf.contrib.layers.instance_norm(x, center=False, scale=False, epsilon=1e-05, trainable=False)
        x = tf.nn.relu(x)

        x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT')
        x = self.conv3(x)
        x = tf.contrib.layers.instance_norm(x, center=False, scale=False, epsilon=1e-05, trainable=False)
        x = tf.nn.tanh(x)
        return x


class Generator(tf.keras.Model):

    def __init__(self, ngf=32, img_size=256, skip=False):
        super(Generator, self).__init__()

        self.img_size = img_size
        self.skip = skip #TODO: Add skip
        self.encoder = Encoder(ngf)
        if(self.img_size == 128):
            self.res1 = Residual(ngf)
            self.res2 = Residual(ngf)
            self.res3 = Residual(ngf)
            self.res4 = Residual(ngf)
            self.res5 = Residual(ngf)
            self.res6 = Residual(ngf)
        else:
            self.res1 = Residual(ngf)
            self.res2 = Residual(ngf)
            self.res3 = Residual(ngf)
            self.res4 = Residual(ngf)
            self.res5 = Residual(ngf)
            self.res6 = Residual(ngf)
            self.res7 = Residual(ngf)
            self.res8 = Residual(ngf)
            self.res9 = Residual(ngf)
        self.decoder = Decoder(ngf)

    @tf.contrib.eager.defun
    def call(self, inputs):
        x = self.encoder(inputs)
        if(self.img_size == 128):
            x = self.res1(x)
            x = self.res2(x)
            x = self.res3(x)
            x = self.res4(x)
            x = self.res5(x)
            x = self.res6(x)
        else:
            x = self.res1(x)
            x = self.res2(x)
            x = self.res3(x)
            x = self.res4(x)
            x = self.res5(x)
            x = self.res6(x)
            x = self.res7(x)
            x = self.res8(x)
            x = self.res9(x)
        x = self.decoder(x)
        return x

class Discriminator(tf.keras.Model):

    def __init__(self, ndf=64):
        super(Discriminator, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(ndf, kernel_size=4, strides=2, padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.conv2 = tf.keras.layers.Conv2D(ndf * 2, kernel_size=4, strides=2, padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.conv3 = tf.keras.layers.Conv2D(ndf * 4, kernel_size=4, strides=2, padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.conv4 = tf.keras.layers.Conv2D(ndf * 8, kernel_size=4, strides=1, padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.conv5 = tf.keras.layers.Conv2D(1, kernel_size=4, strides=1, padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.leaky = tf.keras.layers.LeakyReLU(0.2)

    @tf.contrib.eager.defun
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.leaky(x)

        x = self.conv2(x)
        x = tf.contrib.layers.instance_norm(x, center=False, scale=False, epsilon=1e-05, trainable=False)
        x = self.leaky(x)

        x = self.conv3(x)
        x = tf.contrib.layers.instance_norm(x, center=False, scale=False, epsilon=1e-05, trainable=False)
        x = self.leaky(x)

        x = self.conv4(x)
        x = tf.contrib.layers.instance_norm(x, center=False, scale=False, epsilon=1e-05, trainable=False)
        x = self.leaky(x)

        x = self.conv5(x)
        #x = tf.nn.sigmoid(x) # use_sigmoid = not use_lsgan TODO
        return x
