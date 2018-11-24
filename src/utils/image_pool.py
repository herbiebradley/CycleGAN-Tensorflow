import random

class ImagePool():
    """History of generated images.
    Same logic as https://github.com/junyanz/CycleGAN/blob/master/util/image_pool.lua
    See section 2.3 of https://arxiv.org/pdf/1612.07828.pdf

    Args:
        pool_size (Integer): Max number of images in the history buffer.

    Attributes:
        images (List): List of image batches used to calculate average loss.
        pool_size

    """
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.images = []

    def query(self, image_batch):
        if self.pool_size == 0:
            return image_batch

        if len(self.images) < self.pool_size:
            self.images.append(image_batch)
            return image_batch
        else:
            p = random.random()
            if p > 0.5:
                random_id = random.randrange(0, self.pool_size)
                tmp = self.images[random_id].copy() # Clone here?
                self.images[random_id] = image_batch.copy()
                return tmp
            else:
                return image_batch
# Fix: this does not properly randomly replace images when batch size > 1,
# it does not properly sample min(1, batch_size/2) images
# The order of doing this in training is fine as is
# Check original cyclegan implementation for better code?
