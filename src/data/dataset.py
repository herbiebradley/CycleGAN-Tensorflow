import os

import tensorflow as tf

class Dataset(object):

    def __init__(self, opt):
        self.opt = opt
        self.gpu_id = "/gpu:" + str(self.opt.gpu_id)
        if opt.training:
            self.trainA_path = os.path.join(self.opt.data_dir, 'trainA')
            self.trainB_path = os.path.join(self.opt.data_dir, 'trainB')
            self.trainA_size = len(os.listdir(self.trainA_path))
            self.trainB_size = len(os.listdir(self.trainB_path))
            dataA, dataB = self.load_train_data()
        else:
            self.testA_path = os.path.join(self.opt.data_dir, 'testA')
            self.testB_path = os.path.join(self.opt.data_dir, 'testB')
            self.testA_size = len(os.listdir(self.testA_path))
            self.testB_size = len(os.listdir(self.testB_path))
            dataA, dataB = self.load_test_data()
        self.data = {"A": dataA, "B": dataB}

    def load_train_data(self):
        # Create Dataset from folder of string filenames.
        train_datasetA = tf.data.Dataset.list_files(self.trainA_path + os.sep + '*.jpg', shuffle=False)
        train_datasetB = tf.data.Dataset.list_files(self.trainB_path + os.sep + '*.jpg', shuffle=False)
        # Infinitely loop the dataset, shuffling once per epoch (in memory).
        # Safe to do since the dataset pipeline is currently string filenames.
        # Fused operation is faster than separated shuffle and repeat.
        train_datasetA = train_datasetA.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=self.trainA_size))
        train_datasetB = train_datasetB.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=self.trainB_size))
        # Decodes filenames into jpegs, then stacks them into batches.
        # Throwing away the remainder allows the pipeline to report a fixed sized
        # batch size, aiding in model definition downstream.
        train_datasetA = train_datasetA.apply(tf.contrib.data.map_and_batch(lambda x: self.load_image(x),
                                                                            batch_size=self.opt.batch_size,
                                                                            num_parallel_calls=self.opt.num_threads,
                                                                            drop_remainder=True))
        train_datasetB = train_datasetB.apply(tf.contrib.data.map_and_batch(lambda x: self.load_image(x),
                                                                            batch_size=self.opt.batch_size,
                                                                            num_parallel_calls=self.opt.num_threads,
                                                                            drop_remainder=True))
        # Queue up a number of batches on CPU side:
        train_datasetA = train_datasetA.prefetch(buffer_size=self.opt.num_threads)
        train_datasetB = train_datasetB.prefetch(buffer_size=self.opt.num_threads)
        # Queue up batches asynchronously onto the GPU.
        # As long as there is a pool of batches CPU side a GPU prefetch of 1 is fine.
        # If no GPU exists gpu_id = -1:
        if self.opt.gpu_id != -1:
            train_datasetA = train_datasetA.apply(tf.contrib.data.prefetch_to_device(self.gpu_id, buffer_size=1))
            train_datasetB = train_datasetB.apply(tf.contrib.data.prefetch_to_device(self.gpu_id, buffer_size=1))
        # Create a tf.data.Iterator from the Datasets:
        return iter(train_datasetA), iter(train_datasetB)

    def load_test_data(self):
        test_datasetA = tf.data.Dataset.list_files(self.testA_path + os.sep + '*.jpg', shuffle=False)
        test_datasetB = tf.data.Dataset.list_files(self.testB_path + os.sep + '*.jpg', shuffle=False)
        test_datasetA = test_datasetA.apply(tf.contrib.data.map_and_batch(lambda x: self.load_image(x),
                                                                          batch_size=1,
                                                                          num_parallel_calls=self.opt.num_threads,
                                                                          drop_remainder=False))
        test_datasetB = test_datasetB.apply(tf.contrib.data.map_and_batch(lambda x: self.load_image(x),
                                                                          batch_size=1,
                                                                          num_parallel_calls=self.opt.num_threads,
                                                                          drop_remainder=False))
        test_datasetA = test_datasetA.prefetch(buffer_size=self.opt.num_threads)
        test_datasetB = test_datasetB.prefetch(buffer_size=self.opt.num_threads)
        if self.opt.gpu_id != -1:
            train_datasetA = train_datasetA.apply(tf.contrib.data.prefetch_to_device(self.gpu_id, buffer_size=1))
            train_datasetB = train_datasetB.apply(tf.contrib.data.prefetch_to_device(self.gpu_id, buffer_size=1))
        return iter(test_datasetA), iter(test_datasetB)

    def load_image(self, image_file):
        # Read file into tensor of type string.
        image_string = tf.read_file(image_file)
        # Decodes file into jpg of type uint8 (range [0, 255]).
        image = tf.image.decode_jpeg(image_string, channels=3)
        # Convert to floating point with 32 bits (range [0, 1]).
        image = tf.image.convert_image_dtype(image, tf.float32)
        # Resize with bicubic interpolation, making sure that corner pixel values
        # are preserved.
        image = tf.image.resize_images(image, size=[self.opt.img_size, self.opt.img_size],
                                       method=tf.image.ResizeMethod.BICUBIC, align_corners=True)
        # Transform image to [-1, 1] from [0, 1].
        image = (image - 0.5) * 2
        return image

    def save_images(self, test_images, image_index):
        image_paths = [(os.path.join(opt.results_dir, 'generatedA', 'test' + str(image_index) + '_real.jpg'),
                        os.path.join(opt.results_dir, 'generatedA', 'test' + str(image_index) + '_fake.jpg'),
                        os.path.join(opt.results_dir, 'generatedB', 'test' + str(image_index) + '_real.jpg'),
                        os.path.join(opt.results_dir, 'generatedB', 'test' + str(image_index) + '_fake.jpg')]
        for i in range(len(test_images)):
            # Reshape to get rid of batch size dimension in the tensor.
            image = tf.reshape(test_images[i], shape=[self.opt.img_size, self.opt.img_size, 3])
            # Scale from [-1, 1] to [0, 1).
            image = (image * 0.5) + 0.5
            # Convert to uint8 (range [0, 255]), saturate to avoid possible under/overflow.
            image = tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)
            # JPEG encode image into string Tensor.
            image_string = tf.image.encode_jpeg(image, format='rgb', quality=95)
            tf.write_file(filename=image_paths[i], contents=image_string)

    def get_batches_per_epoch(self, opt):
        # floor(Avg dataset size / batch_size)
        batches_per_epoch = (self.trainA_size + self.trainB_size) // (2 * opt.batch_size)
        return batches_per_epoch
