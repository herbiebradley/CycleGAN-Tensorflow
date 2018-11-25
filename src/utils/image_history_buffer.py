from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

class ImageHistoryBuffer(object):
    """History of generated images.
    Same logic as https://github.com/mjdietzx/SimGAN/blob/master/utils/image_history_buffer.py
    See section 2.3 of https://arxiv.org/pdf/1612.07828.pdf

    Args:
        max_buffer_size (Integer): Max number of images in the history buffer.
        batch_size (Integer): Number of images in incoming batch.
        image_size (Integer): Number of pixels in the width and height dimension of
            incoming images.

    Attributes:
        image_history_buffer (ndarray): Numpy array of image batches used to calculate average loss.

    """
    def __init__(self, max_buffer_size, batch_size, image_size):
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        self.image_history_buffer = np.zeros(shape=(0, image_size, image_size, 3))

    def query(self, image_batch):
        _add_to_image_history_buffer(image_batch)

    def _add_to_image_history_buffer(self, image_batch):
        """Private method to

        Args:
            image_batch (type): Description of parameter `image_batch`.

        """
        images_to_add = max(1, self.batch_size // 2)

        if len(self.image_history_buffer) < self.max_buffer_size:
            self.image_history_buffer = np.append(self.image_history_buffer, image_batch[:images_to_add], axis=0)
        elif len(self.image_history_buffer) == self.max_buffer_size:
            self.image_history_buffer[:images_to_add] = image_batch[:images_to_add]

        np.random.shuffle(self.image_history_buffer)

    def _get_from_image_history_buffer(self):
        """Private method to
        """
        images_to_get = max(1, self.batch_size // 2)
        try:
            return self.image_history_buffer[:images_to_get]
        except IndexError:
            return np.zeros(shape=0)
