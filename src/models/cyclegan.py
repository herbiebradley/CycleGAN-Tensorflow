import tensorflow as tf
import numpy as np
tf.enable_eager_execution()

import os
import time
import glob
import matplotlib.pyplot as plt
import PIL
#from IPython import display

"""### Define Hyperparameters"""

learning_rate = 0.0002
batch_size = 1 ## Set batch size to 4 or 16 if training multigpu
img_size = 256
cyc_lambda = 10
epochs = 1

"""### Load Datasets"""

def load_image(image_file, is_train):
  image = tf.read_file(image_file)
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.cast(image, tf.float32)
  image = tf.image.resize_images(image, [256, 256])
  image = (image / 127.5) - 1
  #if is_train:
    # random jittering

    # resizing to 286 x 286 x 3
    #input_image = tf.image.resize_images(input_image, [286, 286],
                                        #align_corners=True,
                                        #method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #real_image = tf.image.resize_images(real_image, [286, 286],
                                        #align_corners=True,
                                        #method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # randomly cropping to 256 x 256 x 3
    #stacked_image = tf.stack([input_image, real_image], axis=0)
    #cropped_image = tf.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
    #input_image, real_image = cropped_image[0], cropped_image[1]

    #if np.random.random() > 0.5:
      # random mirroring
     # input_image = tf.image.flip_left_right(input_image)
     # real_image = tf.image.flip_left_right(real_image)
  #else:
    #input_image = tf.image.resize_images(input_image, size=[IMG_HEIGHT, IMG_WIDTH],
     #                                    align_corners=True, method=2)
    #real_image = tf.image.resize_images(real_image, size=[IMG_HEIGHT, IMG_WIDTH],
     #                                   align_corners=True, method=2)

  # normalizing the images to [-1, 1]
  #input_image = (input_image / 127.5) - 1
  #real_image = (real_image / 127.5) - 1

  return image

def load_data(batch_size=batch_size):
    path_to_zip = tf.keras.utils.get_file('horse2zebra.zip',
                                          cache_subdir=os.path.abspath('.'),
                                          origin='https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip',
                                          extract=True)

    PATH = os.path.join(os.path.dirname(path_to_zip), 'horse2zebra/')

    train_datasetA = tf.data.Dataset.list_files(PATH+'trainA/*.jpg')
    train_datasetA = train_datasetA.shuffle(1067)
    train_datasetA = train_datasetA.map(lambda x: load_image(x, True))
    train_datasetA = train_datasetA.batch(batch_size) # Repeat() here?
    train_datasetA = iter(train_datasetA)

    train_datasetB = tf.data.Dataset.list_files(PATH+'trainB/*.jpg')
    train_datasetB = train_datasetB.shuffle(1334)
    train_datasetB = train_datasetB.map(lambda x: load_image(x, True))
    train_datasetB = train_datasetB.batch(batch_size) # Repeat() here?
    train_datasetB = iter(train_datasetB)

    return train_datasetA, train_datasetB

"""### Define CycleGAN architecture"""

class Encoder(tf.keras.Model):

  def __init__(self):
    super(Encoder, self).__init__()

    # Small variance in initialization helps with
    self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=7, strides=1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
    self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
    self.conv3 = tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

  def call(self, inputs, training=True):

    x = tf.pad(inputs, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")

    x = self.conv1(x)
    x = tf.contrib.layers.instance_norm(x, epsilon=1e-05, trainable=training) # Implement instance norm to more closely match orig. paper (momentum=0.1)?
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

    x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")

    x = self.conv3(x)
    x = tf.contrib.layers.instance_norm(x, epsilon=1e-05, trainable=training)
    x = tf.nn.tanh(x) # Add 1 and multiply by 127.5 to put img in range [0, 255]?

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
    #x = tf.nn.sigmoid(x) # use_sigmoid = not lsgan

    return x

"""### Define Loss functions"""

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

def cycle_consistency_loss(data_A, data_B, reconstructed_data_A, reconstructed_data_B, cyc_lambda=10):

  loss = tf.reduce_mean(tf.abs(data_A - reconstructed_data_A) + tf.abs(data_B - reconstructed_data_B))

  return cyc_lambda * loss

def generate_images(fake_A, fake_B):
  # the training=True is intentional here since
  # we want the batch statistics while running the model
  # on the test dataset. If we use training=False, we will get
  # the accumulated statistics learned from the training dataset
  # (which we don't want)
  plt.figure(figsize=(15,15))
  fake_A = tf.reshape(fake_A, [256, 256, 3])
  fake_B = tf.reshape(fake_B, [256, 256, 3])
  display_list = [fake_A, fake_B]
  title = ['Generated A', 'Generated B']
  for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()

def define_model(learning_rate):
    discA = Discriminator()
    discB = Discriminator()
    genA2B = Generator()
    genB2A = Generator()

    discA_opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
    discB_opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
    genA2B_opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
    genB2A_opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5)

    model_dict = {"discA":discA, "discB":discB, "genA2B":genA2B, "genB2A":genB2A}
    optim_dict = {"discA_opt":discA_opt, "discB_opt":discB_opt, "genA2B_opt":genA2B_opt, "genB2A_opt":genB2A_opt}
    return model_dict, optim_dict

def train(train_datasetA, train_datasetB, epochs, lsgan=True, learning_rate=learning_rate, cyc_lambda=10):
  model_dict, optim_dict = define_model(learning_rate=learning_rate)

  discA = model_dict["discA"]
  discB = model_dict["discB"]
  genA2B = model_dict["genA2B"]
  genB2A = model_dict["genB2A"]

  discA_opt = optim_dict["discA_opt"]
  discB_opt = optim_dict["discB_opt"]
  genA2B_opt = optim_dict["genA2B_opt"]
  genB2A_opt = optim_dict["genB2A_opt"]

  for epoch in range(epochs):
    start = time.time()

    for i in range(1334):

      with tf.GradientTape() as genA2B_tape, tf.GradientTape() as genB2A_tape, \
           tf.GradientTape() as discA_tape, tf.GradientTape() as discB_tape:
        try:
          # Next training minibatches, default size 1
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

        # Use history buffer of 50 for disc loss
        discA_loss = discriminator_loss(discA_real_output, discA_fake_output, lsgan=lsgan)
        discB_loss = discriminator_loss(discB_real_output, discB_fake_output, lsgan=lsgan)

        genA2B_loss = generator_loss(discB_fake_output, lsgan=lsgan) + \
                      cycle_consistency_loss(trainA, trainB, reconstructedA, reconstructedB, cyc_lambda=cyc_lambda)
        genB2A_loss = generator_loss(discA_fake_output, lsgan=lsgan) + \
                      cycle_consistency_loss(trainA, trainB, reconstructedA, reconstructedB, cyc_lambda=cyc_lambda)

      genA2B_gradients = genA2B_tape.gradient(genA2B_loss, genA2B.variables)
      genB2A_gradients = genB2A_tape.gradient(genB2A_loss, genB2A.variables)

      discA_gradients = discA_tape.gradient(discA_loss, discA.variables)
      discB_gradients = discB_tape.gradient(discB_loss, discB.variables)

      genA2B_opt.apply_gradients(zip(genA2B_gradients, genA2B.variables))
      genB2A_opt.apply_gradients(zip(genB2A_gradients, genB2A.variables))

      discA_opt.apply_gradients(zip(discA_gradients, discA.variables))
      discB_opt.apply_gradients(zip(discB_gradients, discB.variables))

      print("Training step: ", i)

    # saving (checkpoint) the model every 20 epochs
    #if (epoch + 1) % 20 == 0:
      #checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))

if __name__ == "__main__":
    train_datasetA, train_datasetB = load_data(batch_size=batch_size)
    train(train_datasetA, train_datasetB, epochs=1, lsgan=True, learning_rate=learning_rate, cyc_lambda=cyc_lambda)
