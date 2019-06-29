import tensorflow as tf

"""
This file defines the CycleGAN generator and discriminator.
Options are included for extra skips, instance norm, dropout, and resize conv instead of deconv
"""
class Encoder(tf.keras.Model):

    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.use_dropout = opt.use_dropout
        self.norm = opt.instance_norm
        self.training = opt.training
        if self.use_dropout:
            self.norm = False # We don't want to combine instance normalisation and dropout.
            self.dropout = tf.keras.layers.Dropout(opt.dropout_prob)
        self.conv1 = tf.keras.layers.Conv2D(opt.ngf, kernel_size=7, strides=1,
                                            kernel_initializer=tf.truncated_normal_initializer(stddev=opt.init_scale))
        self.conv2 = tf.keras.layers.Conv2D(opt.ngf * 2, kernel_size=3, strides=2, padding='same',
                                            kernel_initializer=tf.truncated_normal_initializer(stddev=opt.init_scale))
        self.conv3 = tf.keras.layers.Conv2D(opt.ngf * 4, kernel_size=3, strides=2, padding='same',
                                            kernel_initializer=tf.truncated_normal_initializer(stddev=opt.init_scale))

    def call(self, inputs):
        # Reflection padding is used to reduce artifacts.
        x = tf.pad(inputs, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT')
        x = self.conv1(x)
        if self.norm:
            x = tf.contrib.layers.instance_norm(x, center=False, scale=False, epsilon=1e-05, trainable=False)
        x = tf.nn.relu(x)
        if self.use_dropout:
            x = self.dropout(x, training=self.training)

        x = self.conv2(x)
        if self.norm:
            x = tf.contrib.layers.instance_norm(x, center=False, scale=False, epsilon=1e-05, trainable=False)
        x = tf.nn.relu(x)
        if self.use_dropout:
            x = self.dropout(x, training=self.training)

        x = self.conv3(x)
        if self.norm:
            x = tf.contrib.layers.instance_norm(x, center=False, scale=False, epsilon=1e-05, trainable=False)
        x = tf.nn.relu(x)
        if self.use_dropout:
            x = self.dropout(x, training=self.training)
        return x


class Residual(tf.keras.Model):

    def __init__(self, opt):
        super(Residual, self).__init__()
        self.use_dropout = opt.use_dropout
        self.norm = opt.instance_norm
        self.training = opt.training
        if self.use_dropout:
            self.norm = False
            self.dropout = tf.keras.layers.Dropout(opt.dropout_prob)
        self.conv1 = tf.keras.layers.Conv2D(opt.ngf * 4, kernel_size=3, strides=1,
                                            kernel_initializer=tf.truncated_normal_initializer(stddev=opt.init_scale))
        self.conv2 = tf.keras.layers.Conv2D(opt.ngf * 4, kernel_size=3, strides=1,
                                            kernel_initializer=tf.truncated_normal_initializer(stddev=opt.init_scale))

    def call(self, inputs):
        x = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        x = self.conv1(x)
        if self.norm:
            x = tf.contrib.layers.instance_norm(x, center=False, scale=False, epsilon=1e-05, trainable=False)
        x = tf.nn.relu(x)
        if self.use_dropout:
            x = self.dropout(x, training=self.training)

        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        x = self.conv2(x)
        if self.norm:
            x = tf.contrib.layers.instance_norm(x, center=False, scale=False, epsilon=1e-05, trainable=False)
        if self.use_dropout:
            x = self.dropout(x, training=self.training)

        x = tf.add(x, inputs) # Add is better than concatenation.
        return x


class Decoder(tf.keras.Model):

    def __init__(self, opt):
        super(Decoder, self).__init__()
        self.use_dropout = opt.use_dropout
        self.norm = opt.instance_norm
        self.training = opt.training
        self.resize_conv = opt.resize_conv
        if self.use_dropout:
            self.norm = False
            self.dropout = tf.keras.layers.Dropout(opt.dropout_prob)
        if self.resize_conv:
            # Nearest neighbour upsampling:
            self.upsample = tf.keras.layers.UpSampling2D(size=(2, 2))
            self.conv1 = tf.keras.layers.Conv2D(opt.ngf * 2, kernel_size=3, strides=1,
                                                kernel_initializer=tf.truncated_normal_initializer(stddev=opt.init_scale))
            self.conv2 = tf.keras.layers.Conv2D(opt.ngf, kernel_size=3, strides=1,
                                                kernel_initializer=tf.truncated_normal_initializer(stddev=opt.init_scale))
        else:
            self.conv1 = tf.keras.layers.Conv2DTranspose(opt.ngf * 2, kernel_size=3, strides=2, padding='same',
                                                kernel_initializer=tf.truncated_normal_initializer(stddev=opt.init_scale))
            self.conv2 = tf.keras.layers.Conv2DTranspose(opt.ngf, kernel_size=3, strides=2, padding='same',
                                                kernel_initializer=tf.truncated_normal_initializer(stddev=opt.init_scale))
        self.conv3 = tf.keras.layers.Conv2D(3, kernel_size=7, strides=1,
                                                kernel_initializer=tf.truncated_normal_initializer(stddev=opt.init_scale))

    def call(self, inputs):
        x = inputs
        if self.resize_conv:
            x = self.upsample(x)
            x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        x = self.conv1(x)
        if self.norm:
            x = tf.contrib.layers.instance_norm(x, center=False, scale=False, epsilon=1e-05, trainable=False)
        x = tf.nn.relu(x)
        if self.use_dropout:
            x = self.dropout(x, training=self.training)

        if self.resize_conv:
            x = self.upsample(x)
            x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        x = self.conv2(x)
        if self.norm:
            x = tf.contrib.layers.instance_norm(x, center=False, scale=False, epsilon=1e-05, trainable=False)
        x = tf.nn.relu(x)
        if self.use_dropout:
            x = self.dropout(x, training=self.training)

        x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT')
        x = self.conv3(x)
        if self.norm:
            x = tf.contrib.layers.instance_norm(x, center=False, scale=False, epsilon=1e-05, trainable=False)
        x = tf.nn.tanh(x)
        if self.use_dropout:
            x = self.dropout(x, training=self.training)
        return x


class Generator(tf.keras.Model):

    def __init__(self, opt):
        super(Generator, self).__init__()
        self.img_size = opt.img_size
        # If true, adds skip connection from the end of the encoder to start of decoder:
        self.gen_skip = opt.gen_skip
        self.encoder = Encoder(opt)
        if(self.img_size == 128):
            self.res1 = Residual(opt)
            self.res2 = Residual(opt)
            self.res3 = Residual(opt)
            self.res4 = Residual(opt)
            self.res5 = Residual(opt)
            self.res6 = Residual(opt)
        else:
            self.res1 = Residual(opt)
            self.res2 = Residual(opt)
            self.res3 = Residual(opt)
            self.res4 = Residual(opt)
            self.res5 = Residual(opt)
            self.res6 = Residual(opt)
            self.res7 = Residual(opt)
            self.res8 = Residual(opt)
            self.res9 = Residual(opt)
        self.decoder = Decoder(opt)

    @tf.contrib.eager.defun
    def call(self, inputs):
        inputs = self.encoder(inputs)
        if(self.img_size == 128):
            x = self.res1(inputs)
            x = self.res2(x)
            x = self.res3(x)
            x = self.res4(x)
            x = self.res5(x)
            x = self.res6(x)
            if(self.gen_skip):
                x = tf.add(x, inputs)
        else:
            x = self.res1(inputs)
            x = self.res2(x)
            x = self.res3(x)
            x = self.res4(x)
            x = self.res5(x)
            x = self.res6(x)
            x = self.res7(x)
            x = self.res8(x)
            x = self.res9(x)
            if(self.gen_skip):
                x = tf.add(x, inputs)
        x = self.decoder(x)
        return x


class Discriminator(tf.keras.Model):

    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.norm = opt.instance_norm
        self.conv1 = tf.keras.layers.Conv2D(opt.ndf, kernel_size=4, strides=2, padding='same',
                                            kernel_initializer=tf.truncated_normal_initializer(stddev=opt.init_scale))
        self.conv2 = tf.keras.layers.Conv2D(opt.ndf * 2, kernel_size=4, strides=2, padding='same',
                                            kernel_initializer=tf.truncated_normal_initializer(stddev=opt.init_scale))
        self.conv3 = tf.keras.layers.Conv2D(opt.ndf * 4, kernel_size=4, strides=2, padding='same',
                                            kernel_initializer=tf.truncated_normal_initializer(stddev=opt.init_scale))
        self.conv4 = tf.keras.layers.Conv2D(opt.ndf * 8, kernel_size=4, strides=1, padding='same',
                                            kernel_initializer=tf.truncated_normal_initializer(stddev=opt.init_scale))
        self.conv5 = tf.keras.layers.Conv2D(1, kernel_size=4, strides=1, padding='same',
                                            kernel_initializer=tf.truncated_normal_initializer(stddev=opt.init_scale))
        self.leaky = tf.keras.layers.LeakyReLU(0.2)

    @tf.contrib.eager.defun
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.leaky(x)

        x = self.conv2(x)
        if self.norm:
            x = tf.contrib.layers.instance_norm(x, center=False, scale=False, epsilon=1e-05, trainable=False)
        x = self.leaky(x)

        x = self.conv3(x)
        if self.norm:
            x = tf.contrib.layers.instance_norm(x, center=False, scale=False, epsilon=1e-05, trainable=False)
        x = self.leaky(x)

        x = self.conv4(x)
        if self.norm:
            x = tf.contrib.layers.instance_norm(x, center=False, scale=False, epsilon=1e-05, trainable=False)
        x = self.leaky(x)

        x = self.conv5(x)
        #x = tf.nn.sigmoid(x) # use_sigmoid = not use_lsgan TODO
        return x
