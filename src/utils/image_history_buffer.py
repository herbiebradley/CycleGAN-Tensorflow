from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import tensorflow as tf
import numpy as np

class ImageHistoryBuffer(object):
    """History of generated images.
    Similar logic to https://github.com/mjdietzx/SimGAN/blob/master/utils/image_history_buffer.py
    See section 2.3 of https://arxiv.org/pdf/1612.07828.pdf

    Args:
        max_buffer_size (Integer): Max number of images in the history buffer.
        batch_size (Integer): Number of images in incoming batch.
        img_size (Integer): Number of pixels in the width and height dimension of
            incoming images.

    Attributes:
        image_history_buffer: Tensor of image batches used to calculate average loss.

    """
    def __init__(self, max_buffer_size, batch_size, img_size):
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        self.image_history_buffer = np.zeros(shape=(0, img_size, img_size, 3))
        assert(self.batch_size >= 1)

    def query(self, image_batch):
        """Adds max(1, batch size / 2) images from the incoming batch to the
            history buffer, then randomly replaces max(1, batch size / 2) images
            in the batch with images sampled from the buffer. If batch size is 1,
            then we flip a coin to decide if we return a random image from the
            buffer or the original image.

        Args:
            image_batch: Tensor of shape=(batch_size, img_size, img_size, 3).

        Returns:
            Tensor: Processed batch.

        """

        image_batch = image_batch.numpy()
        self._add_to_image_history_buffer(image_batch)
        if self.batch_size > 1:
            image_batch[:self.batch_size // 2] = self._get_from_image_history_buffer()
        else:
            p = random.random()
            if p > 0.5:
                random_image = self._get_from_image_history_buffer()
                return tf.convert_to_tensor(random_image, dtype=tf.float32)
        return tf.convert_to_tensor(image_batch, dtype=tf.float32)

    def _add_to_image_history_buffer(self, image_batch):
        """Private method to add max(1, batch size / 2) images to buffer.

        Args:
            image_batch (Tensor): Incoming image batch.

        """
        images_to_add = max(1, self.batch_size // 2)

        if len(self.image_history_buffer) < self.max_buffer_size:
            self.image_history_buffer = np.append(self.image_history_buffer, image_batch[:images_to_add], axis=0)
        else:
            self.image_history_buffer[:images_to_add] = image_batch[:images_to_add]

        np.random.shuffle(self.image_history_buffer)

    def _get_from_image_history_buffer(self):
        """Private method to get random images from buffer. The randomness is
            achieved with the shuffle in the _add_to method, since _add_to and
            _get_from are always called consecutively.
        """
        images_to_get = max(1, self.batch_size // 2)
        return self.image_history_buffer[:images_to_get]
