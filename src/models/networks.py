from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class Encoder(tf.keras.Model):

    def __init__(self):
        super(Encoder, self).__init__()
        # Small variance in initialization helps with preventing colour inversion.
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=7, strides=1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.conv3 = tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

    def call(self, inputs, training=True):
        x = tf.pad(inputs, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT')
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
        x = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        x = self.conv1(x)
        x = tf.contrib.layers.instance_norm(x, epsilon=1e-05, trainable=training)
        x = tf.nn.relu(x)

        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        x = self.conv2(x)
        x = tf.contrib.layers.instance_norm(x, epsilon=1e-05, trainable=training)

        x = tf.add(x, inputs) # Consider concatenating in channel dimension here for better net
        return x


class Decoder(tf.keras.Model):

    def __init__(self):
        super(Decoder, self).__init__()

        self.conv1 = tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.conv2 = tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.conv3 = tf.keras.layers.Conv2D(3, kernel_size=7, strides=1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

    def call(self, inputs, training=True):
        x = self.conv1(inputs)
        x = tf.contrib.layers.instance_norm(x, epsilon=1e-05, trainable=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = tf.contrib.layers.instance_norm(x, epsilon=1e-05, trainable=training)
        x = tf.nn.relu(x)

        x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT')
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
        if(self.img_size == 128):
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
        if(self.img_size == 128):
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
        # TODO: check padding here, should it be same?
        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.conv2 = tf.keras.layers.Conv2D(128, kernel_size=4, strides=2, padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.conv3 = tf.keras.layers.Conv2D(256, kernel_size=4, strides=2, padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.conv4 = tf.keras.layers.Conv2D(512, kernel_size=4, strides=1, padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.conv5 = tf.keras.layers.Conv2D(1, kernel_size=4, strides=1, padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
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
        #x = tf.nn.sigmoid(x) # use_sigmoid = not use_lsgan
        return x
